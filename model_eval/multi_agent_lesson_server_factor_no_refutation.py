from client.models import LLM4PP_Problem, LLM4PP_Submission
from client.pareval_client import ParEvalDriver
from client.polybench_client import PolyBenchDriver
from typing import List, Dict, Type
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModel
import json
import torch
import numpy as np
import os
from scipy.stats import entropy
import time
from openai import OpenAI
import random
import math
import logging
from logging import Logger
import hydra
import jsonlines
from omegaconf import DictConfig, OmegaConf



from debate.utils import *
from debate.agent_vllm import *

# Load the model and move it to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute magnitudes (L2 norms)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)

def prompt_truncation(prompt: str, length: int=9000) -> str:
                #print("prompt:")
    #print("before: ", len(prompt))
    if len(prompt) > length:
        prompt = prompt[:length]
    #print("after: ", len(prompt))
    return prompt


def get_embedding(tokenizer, model, input: str):
    # Tokenize the explanation
    inputs = tokenizer(input, return_tensors="pt", truncation=True, padding=True).to(device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()

    return cls_embedding

def misconception_refutation(lessons_codes: List[Dict[str, any]], agent_list: List[VLLMAgent], length_tokenizer: AutoTokenizer, logger: Logger) -> list:
    total_in_tokens = 0
    total_out_tokens = 0

    for i in range(len(lessons_codes)):
        lesson_and_code = lessons_codes[i]
        idx = lesson_and_code["idx"]
        random_agent = agent_list[idx] # Not Random for now

        issues, in_tokens, out_tokens = random_agent.identify_lesson(context=agent_list[idx].context)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        issues_length = length_tokenizer.tokenize(issues)
        logger.debug("issues %d length : %d", i, len(issues_length))
        logger.debug("Issues %d Before Summarization: \n%s", i, issues)

        logger.debug("issues length : %d", len(issues_length))
        logger.debug("Issues %d After Summarization: \n%s", i, issues)

        # NOTE: clean issues
        if "The explanation is correct." in issues and not contains_target_words(issues):
            continue
        else:
            if issues.startswith("The explanation is correct."):
                # Remove the target phrase from the beginning of the sentence
                issues = issues[len("The explanation is correct."):].strip()
            logger.info("Refutation Start")
            agent_list[idx].context["issues"] = issues
            modified_lesson, in_tokens, out_tokens = random_agent.modify_lesson(context=agent_list[idx].context)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens
            #logger.info("Modified Lessons %d: \n%s", i, modified_lesson)
            while modified_lesson is None: # Regenerate if there is no explanation, NOTE: not in this case because the modified lesson is not cleaned
                #logger.info("Modified Lessons %d: \n%s", i, modified_lesson)
                modified_lesson, in_tokens, out_tokens = random_agent.modify_lesson(context=agent_list[idx].context)
                total_in_tokens += in_tokens
                total_out_tokens += out_tokens
            modified_lesson_length = length_tokenizer.tokenize(modified_lesson)
            logger.info("modified_lesson length : %d", len(modified_lesson_length))
            logger.debug("Modified Lessons %d Before Summarization: \n%s", i, modified_lesson)

            logger.debug("Modified Lessons %d After Summarization: \n%s", i, modified_lesson)
            lessons_codes[i]["lesson"] = modified_lesson
    return lessons_codes, total_in_tokens, total_out_tokens

def select_high_speedup(all_lessons_codes: List[Dict[str, any]], count: int):
    """
    Given lessons with codes (Z_all), select k/2 of lessons out of the whole Z_all with good speedup (>= 1.1)

    Args:
        all_lessons_codes (List[Dict[str, any]]): Z_all, lessons with codes and speedup
        count (int): k/2, the number of lessons chosen with high speedup
    
    Returns:
        List[Dict[str, any]]: lessons with codes, speedup and other information, where all lessons has high speedup
    """

    lessons_codes_high_speedup = [lesson for lesson in all_lessons_codes if lesson["speedup"] * lesson["factor"] >= 1.1]
    lessons_codes_remain = [lesson for lesson in all_lessons_codes if lesson["speedup"] * lesson["factor"] < 1.1]

    if len(lessons_codes_high_speedup) > count:
        lessons_codes_high_speedup = sorted(lessons_codes_high_speedup, key=lambda x: x["speedup"] * x["factor"], reverse=True)[:count]
    
    return lessons_codes_high_speedup, lessons_codes_remain

def select_high_quality(lessons_codes_remain: List[Dict[str, any]], count: int):
    """
    Given remaining low-speedup lessons with codes (Z_remain), select k/2 of lessons out of the whole Z_remain with good quality by cosine similarity

    Args:
        lessons_codes_remain (List[Dict[str, any]]): Z_remain, low speedup lessons with codes, assume quality is already available for each element of dictionary
        count (int): k/2, the number of lessons chosen with high quality
    
    Returns:
        List[Dict[str, any]]: lessons with codes, speedup and other information, where all lessons has high speedup
    """

    #optimized_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=optimized_code_list)

    lessons_codes_high_quality = sorted(lessons_codes_remain, key=lambda x: x["quality"], reverse=True)[:count]
    return lessons_codes_high_quality

def get_code_speedup_lesson(problem: LLM4PP_Problem, agent_list: List[VLLMAgent], length_tokenizer: AutoTokenizer, logger: Logger, driver: ParEvalDriver, t: int, lesson_count: int, all_lessons_codes: list):
    source_code = problem.source_code
    lessons_codes = [] # lessons and codes with embedding, the current Z
    optimized_code_list = []
    submission_list=[]
    response_list = []
    lesson_list = []
    source_code_list = []
    speedup_list = []
    lesson_counts = agent_list[0].context["lesson_counts"]

    total_in_tokens = 0
    total_out_tokens = 0
    
    # ================= Round 0:  GET Code, Speedup and Lessons ==================================
    logger.info("round: %d", t)
    # Get Code from Agents, corresponding submission and response of generated code, and Lessons of generated code.
    for i in range(len(agent_list)):
        agent = agent_list[i]
        optimized_code, in_tokens, out_tokens = agent.optimize_code(context=agent.context)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        optimized_code_list.append(optimized_code)
        submission = LLM4PP_Submission(problem=problem, submitted_code=optimized_code)
        submission_list.append(optimized_code)
        try:
            response = driver.submit(submission)
        except Exception as e:
            print(f"submission {i+1} error")
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response.stdout)
        response_list.append(response)

        # Generate Lesson
        agent.context["tgt_code"] = response.submission.submitted_code
        agent.context["feedback"] = response.stdout
        lesson, in_tokens, out_tokens = agent.generate_lesson(context=agent.context)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens

        lesson_length = length_tokenizer.tokenize(lesson)
        logger.info("Lesson %d length : %d", i, len(lesson_length))

        lesson_list.append(lesson)
        logger.debug("Lesson %d content: %s\n", i, lesson)

        agent.context["lesson"] = lesson
        source_code_list.append(agent.context["src_code"])

        log, tag, speedup = pareval_process_execution_feedback(response.stdout)

        if tag != "CORRECT":
            speedup = 0
        speedup_list.append(speedup)



        
        lesson_count += 1
        lesson_info_dict = {
            "lesson": lesson,
            "tag": tag,
            "log": log,
            "speedup": speedup,
            "src_code": source_code,
            "tgt_code": optimized_code,
            #"lesson_emb": lesson_emb[i],
            #"tgt_code_emb": source_code_emb[i],
            "idx": i,
            "quality": -1,
            "round": t,
            "factor": 1,
            "lesson_count": lesson_count,
            "used_lessons": agent.context["lessons"]
        }
        lessons_codes.append(lesson_info_dict)
    
    if t > 0:
        for lesson in all_lessons_codes:
            factor = 0
            lesson_count = lesson.get("lesson_count")
            if lesson_count not in lesson_counts:
                continue
            # if lesson_count in lesson_counts:
            prev_speedup = lesson.get("speedup")
            for curr_sp in speedup_list:
                if curr_sp > prev_speedup:
                    factor += 1.1
                else:
                    factor += 0.9
            factor = factor / len(speedup_list)
            lesson["factor"] = factor
    
    return lessons_codes, lesson_count, all_lessons_codes, total_in_tokens, total_out_tokens


@hydra.main(version_base=None, config_path="config", config_name="debate_config")
def main(cfg : DictConfig) -> None:

    benchmark = cfg.benchmark

    if benchmark == "ParEval":
        driver = ParEvalDriver()
        evaldriver = ParEvalDriver()
    elif benchmark == "PolyBench":
        driver = PolyBenchDriver()
        evaldriver = PolyBenchDriver()
    else:
        print("Unknown Benchmark, program exits.")
        exit(0)

    logging.config.dictConfig(cfg.logging)

    # A logger for this file
    logger = logging.getLogger("main_logger")
    class_logger = logging.getLogger("class_logger")

    T = cfg.Rounds # how many rounds of debate for one specific problem
    k = cfg.k # how many lessons to put in agents for learning

    temperature = cfg.temperature
    reason_temperature = cfg.reason_temperature
    mode = cfg.mode

    if mode != "serial": # assume mode is a parallel package can be integrated in c++ only
        additional_package = f"Feel free to use {mode} to parallelize the code."
    else: #serial
        additional_package = ""


    localhost = cfg.localhost
    savename = f"{benchmark}_lesson_factor_{temperature}_{reason_temperature}_no_refutation.jsonl"
    savedir = "logs"
    os.makedirs(savedir, exist_ok=True)
    save_destination = f"{savedir}/{savename}"
    print("save name and destination: ", savename, save_destination)

    # Sampling Parameters
    # sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=2048)
    # reasoning_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024, repetition_penalty=1.5)

    length_tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")


    # Initialize Agents
    llm_1 = OpenAI(
        base_url=f"http://{localhost}:8001/v1",
        api_key="token-0",
    )
    
    agent_1 = VLLMAgent(llm=llm_1, model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)


    llm_2 = OpenAI(
        base_url=f"http://{localhost}:8002/v1",
        api_key="token-1",
    )

    agent_2 = VLLMAgent(llm=llm_2, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)
    llm_3 = OpenAI(
        base_url=f"http://{localhost}:8003/v1",
        api_key="token-2",
    )

    #agent_3 = VLLMAgent(llm=llm_3, model_name="Qwen/CodeQwen1.5-7B-Chat", logger=class_logger)
    #codellama/CodeLlama-13b-Instruct-hf
    #agent_3 = VLLMAgent(llm=llm_3, model_name="bigcode/starcoder2-15b-instruct-v0.1", logger=class_logger)
    #Qwen/Qwen2.5-Coder-14B-Instruct
    #agent_3 = VLLMAgent(llm=llm_3, model_name="meta-llama/Llama-3.1-8B-Instruct", logger=class_logger)
    agent_3 = VLLMAgent(llm=llm_3, model_name="Qwen/Qwen2.5-Coder-14B-Instruct", temperature=temperature, reason_temperature=reason_temperature, length_tokenizer=length_tokenizer, additional_package=additional_package, logger=class_logger)
    agent_list = [agent_1, agent_2, agent_3]

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    model = AutoModel.from_pretrained("neulab/codebert-cpp").to(device)

    lessons_history = []

    start_time = time.time() 
    logger.info("Start Debating")

    count = 0
    lesson_count = 0
    total_in_tokens = 0
    total_out_tokens = 0

    for problem in evaldriver:
        problem : LLM4PP_Problem

        #if problem.problem_id != "13_geometry_closest_pair_2d":
        #    continue
            
        
        #if problem.problem_id != "12_geometry_smallest_triangle":
        #    continue
            
        # if problem.problem_id != "14_geometry_closest_pair_1d":
        #     continue
            
        #if problem.problem_id != "11_geometry_convex_hull_perimeter":
        #    continue
            
        # if problem.problem_id != "10_geometry_convex_hull":
        #     continue

        logger.info(problem.problem_id)
        logger.info(problem.category)
        prev_speedup = 1

        # if problem.category == 'search':
        #     continue

        source_code = problem.source_code
        all_lessons_codes = [] # the Z_all
        # For each problem, initialize the context that is used to prompt LLM
        for agent in agent_list:
            agent.context = {
                "src_code": source_code,
                "tgt_code": "",
                "feedback": "",
                "lesson": "", # the lesson corresponds to the generated code, i.e. tgt_code
                "lessons": "", # the lessons put in the prompt for future rounds
                "issues": "",
                "lesson_counts": [],
                "problem_id": problem.problem_id,
                "category": problem.category
            }
        
        code_to_submit = ""
        

        current_lessons_codes, lesson_count, all_lessons_codes, in_tokens, out_tokens = get_code_speedup_lesson(problem=problem, 
                                                                      agent_list=agent_list, 
                                                                      length_tokenizer=length_tokenizer, 
                                                                      logger=logger, 
                                                                      driver=driver, 
                                                                      t=0, 
                                                                      lesson_count=lesson_count,
                                                                      all_lessons_codes=all_lessons_codes)
        total_in_tokens += in_tokens
        total_out_tokens += out_tokens
        # Perform Misconception Refutation
        # corrected_lessons_codes, in_tokens, out_tokens = misconception_refutation(lessons_codes=current_lessons_codes, agent_list=agent_list, length_tokenizer=length_tokenizer, logger=logger)
        #all_lessons_codes += corrected_lessons_codes

        corrected_lessons_codes = current_lessons_codes
        # total_in_tokens += in_tokens
        # total_out_tokens += out_tokens
        # Get the embedding of source code and corrected lesson, and record the quality score
        # source_code_list = [item["src_code"] for item in corrected_lessons_codes]
        # corrected_lesson_list = [item["lesson"] for item in corrected_lessons_codes]

        source_code_list = [item["src_code"] for item in corrected_lessons_codes]
        corrected_lesson_list = [item["lesson"] for item in corrected_lessons_codes]

        source_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=source_code_list)
        corrected_lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=corrected_lesson_list)

        for i in range(len(corrected_lessons_codes)):
            quality = cosine_similarity(corrected_lesson_emb[i], source_code_emb[i])
            corrected_lessons_codes[i]["quality"] = quality
        
        all_lessons_codes += corrected_lessons_codes
        


        # ================ Round 0 Finished ===============================
        
        for t in range(1, T):
            logger.info("round: %d", t)

            if t * len(agent_list) <= k:
                lessons_codes_next_round = all_lessons_codes
            else:
                lessons_codes_high_speedup, lessons_codes_remain = select_high_speedup(all_lessons_codes, count=math.ceil(k/2))
                num_high_speedup = len(lessons_codes_high_speedup)
                num_high_quality = k - num_high_speedup
                lessons_codes_low_speedup = select_high_quality(lessons_codes_remain, count=num_high_quality)
                lessons_codes_next_round = lessons_codes_high_speedup + lessons_codes_low_speedup
            
            sorted_lessons_codes = sorted(lessons_codes_next_round, key=lambda x: x["speedup"], reverse=True)

            logger.info("Best Speedup After Debate Round %d : %f", t, sorted_lessons_codes[0]["speedup"])

            
            #new_source_code = sorted_lessons_codes[0]["src_code"]
            new_source_code = source_code
            source_code = new_source_code

            lesson_to_agent = [{"lesson": item["lesson"], "tag": item["tag"], "speedup": item["speedup"], "lesson_count": item["lesson_count"]} for item in lessons_codes_next_round]

            # Extract lesson_count corresponding to each lesson in lesson_to_agent
            lesson_counts = [
                item["lesson_count"] for item in lesson_to_agent
            ]
            # Update agent history and source code
            for agent in agent_list:
                agent.single_round_memory.append(agent.context)
                agent.context = {
                "src_code": new_source_code,
                "tgt_code": "",
                "feedback": "",
                "lesson": "",
                "lessons": lesson_to_agent,
                "issues": "",
                "lesson_counts": lesson_counts,
                "problem_id": problem.problem_id,
                "category": problem.category
            }
            
            current_lessons_codes, lesson_count, all_lessons_codes, in_tokens, out_tokens = get_code_speedup_lesson(problem=problem, 
                                                                          agent_list=agent_list, 
                                                                          length_tokenizer=length_tokenizer, 
                                                                          logger=logger, 
                                                                          driver=driver, 
                                                                          t=t,
                                                                          lesson_count=lesson_count,
                                                                          all_lessons_codes=all_lessons_codes)
            total_in_tokens += in_tokens
            total_out_tokens += out_tokens
            # Perform Misconception Refutation
            # corrected_lessons_codes, in_tokens, out_tokens = misconception_refutation(lessons_codes=current_lessons_codes, agent_list=agent_list, length_tokenizer=length_tokenizer, logger=logger)
            # total_in_tokens += in_tokens
            # total_out_tokens += out_tokens

            source_code_list = [item["src_code"] for item in current_lessons_codes]
            current_lesson_list = [item["lesson"] for item in current_lessons_codes]

            source_code_emb = get_embedding(tokenizer=tokenizer, model=model, input=source_code_list)
            current_lesson_emb = get_embedding(tokenizer=tokenizer, model=model, input=current_lesson_list)

            for i in range(len(current_lessons_codes)):
                quality = cosine_similarity(current_lesson_emb[i], source_code_emb[i])
                current_lessons_codes[i]["quality"] = quality
                
            #all_lessons_codes += corrected_lessons_codes
            all_lessons_codes += current_lessons_codes
        
        #last_lessons_codes = all_lessons_codes[-len(agent_list):]
        sorted_lessons_codes = sorted(all_lessons_codes, key=lambda x: x["speedup"], reverse=True)
        logger.info("Best Speedup After Whole Debate: %f", sorted_lessons_codes[0]["speedup"])

        #code_to_submit = sorted_lessons_codes[0]["src_code"] if sorted_lessons_codes[0]["speedup"] >= prev_speedup else source_code # Should be same as prev speedup as updated before
        #NOTE: no baseline source code if we have bad speedup
        code_to_submit = sorted_lessons_codes[0]["tgt_code"]
        submission_best = LLM4PP_Submission(problem=problem, submitted_code=code_to_submit)
        try:
            response_submit = evaldriver.submit(submission_best)
        except Exception as e:
            print(f"skipping problem due to exception: {e}")
            print("--- ParEval driver stdout ---")
            print(response_submit.stdout)
        
        lessons_history.append({
            "lessons": sorted_lessons_codes,
            "problem_id": problem.problem_id,
            "category": problem.category
            }
        )
        for agent in agent_list:
            agent.memory.append(agent.single_round_memory)
            agent.single_round_memory = []
        count += 1


    start_2 = time.time()
    evaldriver.save_all_responses("./lesson-3-4-results.json")
    evaldriver.evaluate()
    start_3 = time.time()

    print("total time: ", start_3 - start_time)
    print("running time: ", start_2 - start_time)
    # Write to JSON Lines file

    print("Total in tokens: ", total_in_tokens)
    print("Total out tokens: ", total_out_tokens)
    price = total_in_tokens * 0.150 / 1000000 + total_out_tokens * 0.6 / 1000000
    print("GPT 4o mini price: ", price)


    # for item in lessons_history:
    #     for item2 in item["lessons"]:

            
    #         if isinstance(item2.get("quality"),np.float32):
    #             item2["quality"] = float(item2["quality"])
            

    with jsonlines.open(save_destination, mode="w") as writer:
        writer.write_all(lessons_history)  # Writes all dictionaries as separate lines
    # with jsonlines.open("debate_lessons_history.jsonl", mode="w") as writer:
    #     writer.write_all(lessons_history)  # Writes all dictionaries as separate lines
    # with open("debate_model_category_prompts_7b.json", "w") as file:
    #     json.dump(reason_category_dict, file, indent=4) 
    # for agent in agent_list:
    #     agent.save_memory()

if __name__ == "__main__":
    main()
