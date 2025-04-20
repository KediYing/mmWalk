#!/usr/bin/env python3
# THIS IS A GENERAL SCRIPT, PLEASE ENSURE YOU HAVE THE DATASET IN RIGHT DIRECTORIES
import os
import json
import argparse
import random
import math
from pathlib import Path
import base64
from openai import OpenAI
import time
from tqdm import tqdm
import re
from collections import defaultdict
from PIL import Image
import io
import numpy as np
import sys
import shutil


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from converter import convert_json_to_descriptions, resolve_path


OPENAI_API_KEY = "JAMES_BOND"  # 请替换为您的密钥

SYSTEM_MESSAGE = """
Generate 15 QA pairs based on multi-view scene graph and json file represented. A FRAME represents one frame of one trajectory, which contains one Json file that describes this certain frame and 3 RGB images under three different views(dog, walker and drone), along with text descriptions of semantic and depth information. Generate Question Answer pairs for each FRAME which should cover all 9 QA Types.(At least 1 pair for each type) The Question Answer pairs must follow the instruction and rules below, taking the given sample as example.
"""

INSTRUCTION_CONTENT = """
Rules:
1.Only describe clear information in the images - do not fabricate or invent in the answers.
2.Base ALL answers ONLY on what is actually visible in the provided rgb images and stated in the JSON data. Do not make assumptions or invent details.
3.ALL Position information must be described in clockwise manner.(Instead of left/right, describe exact clockwise location such as 'your 3 o'clock')


Instructions:
Easy Level QA
QA pairs that query the basic information in the Json file or single image, the answer can be completely verified by the ground truth. Consider the questioner is the pedestrain in the first-person perspective of every scene, use "my surrounding" or current environment instead of "scene" in question.
-Type E1- General Query: the simple query questions about single feature(weather, action)), the answer should be concise and in several words or at most one sentence.
<sample_Question1> What is the weather now?
<sample_Answer1> Rainy.
<sample_Question2> What is the current action?
<sample_Answer2> Turning Left.
-Type E2- Existence: Query existence of specific objects, or corner cases or landmarks in the current rgb image, the answer should be Yes or No with at most one sentence for extension, considering only walker view, dont mention views in question and answers. Questions should mainly based on RGB images, do NOT only ask about landmarks. You MUST have a 25 per cent probability of using a question with an not existing object and with the answer of NO. Do NOT always use same object like 'bench','pedestrain crossing'
<sample_Question1> Is there any benches in the scene?
<sample_Answer1> No.
<sample_Question2> Does the scene contain any corner cases for blind?
<sample_Answer2> Yes, the scene contains multiple "Barriers" and one Narrow Path.
-Type E3- Counting: Inquire the exact number of features, using only walker view, dont mention views in question and answers
<sample_Question1> How many vehicles are there in the parking area?
<sample_Answer1> 3
<sample_Question2> What types of vehicles are there?
<sample_Answer2> The scene contains 2 cars(label14) and 1 bicycle(label19).
-Type E4- Attribute: simple query questions based on the RGB Image(color,shape,size,texture), the answer should be at most two sentences.
<sample_Question1> What is the color of the car in gas station?
<sample_Answer1> Black.
<sample_Question2> How large is the Barrier in front of me?
<sample_Answer2> There are some large cardboard boxes stacked on top of each other on the pavement, 1 meter high and 1 meter wide.

Middle Level QA
QA pairs that contain multiple views or multiple images in concern, the answer stems mainly from the combined ground truth feature information. The answer can be partially verified. Consider the questioner as an analyst instead of the pedestrain.
-Type M1- Spatial: Query about the spatial information (distance,position) between multiple objects.
<sample_Question1> What is the relative position of the sidewalk to the bus stop? 
<sample_Answer1> The bus stop is located on only one sidewalk (label 2), you are currently standing on the side with bus top.
<sample_Question2> How far is the intersection?
<sample_Answer2> The intersection is around 20 meters far.
<sample_Question3> Where is the bus stop according to my position?
<sample_Answer3> The bus stop is 12 meters far at your 4 o'clock.

-Type M2- Description: Describing the scene with all(or many) features depending on the question. Answers must be completely based on the information in multiple features and be at most 3 sentences, depending on the complexity of the scene, rely more on depth information.

-Type M3- Comparison:Compare different three views(dog/walker/drone), answers must be completely relevant to the enquiry and give more detailed reasons only based on ground truth. Answers must be at most 2 sentences, refer the appeared corner case in answer in CONCRETE.
<sample_Question> Which view gives more information about the dangerous environment?
<sample_Answer> The pedestrain is about to cross multiple narrow paths, the obstacles take larger semantic percentage of the frame in the 'dog' view. Whereas the drone view is partially obscured by the foliage as the high obstacle. The dog view therefore provides the most information to ensure your safety.

Hard Level QA:
QA pairs that based on multiple features and multiple frames but require further merging, processing, analyzing and expanding. The question is on the abstract and summarizing perspective. The answer stems mainly from the processed ground truth information and image, refer the appeared corner case if there is any. The answer must be at most 3 sentences long.
-Type H1- Risk Assessment: Calculate the Risk of the scene based the information and Risk Assessment Function. The answer should give the final score based on information in RiskFunction and brief explanation in few sentences, don't mention the Function.
<sample_Question> Is the current scenario safe?
<sample_Answer> The current scenario is rated generally as high-risk, considering the current foggy weather and the need to cross the road ahead without street lights. Foggy days can cause drivers of vehicles to have a lower visible distance and a greater potential threat when crossing the road.
-Type H2- Landmarks and Navigation:Evaluate landmarks in the surroundings based on navigational value, and make an overall evaluation based on the size, distance, and number of landmarks, add concrete but concise reason based on image, not only saying that landmark has fixed position.
<sample_Question> Which landmarks in this scene would be most valuable for a blind pedestrian's navigation, and why?
<sample_Answer> The traffic light pole at 3 meters straight ahead in the direction of your 11 o'clock serves as the most valuable landmark due to its fixed position, distinctive sound, and proximity. The bus stop 8 meters far at your 2 o'clock offers secondary value with its shelter structure and bench, providing both tactile and spatial reference. The storefront signs on the left, while visible, have lower navigational value due to potential changes. Overall, this scene offers moderate landmark quality for navigation, with the pole and bus stop creating a reliable orientation corridor.

QA pairs types are labelled as follows:
- 4 Easy level questions (E1, E2, E3, E4)
- 3 Middle level questions (M1, M2, M3)
- 2 Hard level questions (H1, H2)

Format your response as a JSON array with these QA pairs. Each pair should include:
- question: the question text
- answer: the comprehensive answer
- difficulty: "Easy", "Middle", or "Hard"
- category: the specific type (E1, E2, E3, E4, M1, M2, M3, H1, H2)
"""


IMAGE_DESCRIPTION = """These images show the CURRENT SCENE from three camera positions (walker at eye level, drone from above, dog from low position).

The RGB images show actual colors. Instead of depth and semantic images, we provide text descriptions of objects with their clock positions and approximate distances.

IMPORTANT: When analyzing "Static" objects (immovable elements like fire hydrants, benches, fountains, bus stops) and "Dynamic" objects (elements that can change position like trash bins, bags, wheelchairs, animals), you MUST closely examine the RGB images to identify their specific type, appearance, and characteristics. The semantic labels only tell you that an object belongs to these categories, but the RGB images show what specific objects they actually are.
"""

def format_semantic_depth_descriptions(descriptions):
    formatted_text = "SEMANTIC AND DEPTH DESCRIPTIONS:\n"
    
    view_names = ["Drone", "Walker", "Dog"]
    for i, (view_name, view_desc) in enumerate(zip(view_names, descriptions)):
        formatted_text += f"{view_name} view: "
        if view_desc:
            formatted_text += ", ".join(view_desc)
        else:
            formatted_text += "No objects detected"
        formatted_text += "\n"
    
    return formatted_text

# Downsample to reduce cost
def downsample_image(image_path, max_size=384, quality=70):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            img_mode = img.mode
            
            resize_ratio = min(max_size / width, max_size / height)

            if resize_ratio >= 1:
                new_width, new_height = width, height
            else:
                new_width = int(width * resize_ratio)
                new_height = int(height * resize_ratio)

            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            original_size = os.path.getsize(image_path) / 1024  # KB
            buffer = io.BytesIO()

            if img_mode in ['I', 'I;16', 'F']:
                resized_img = resized_img.convert('L')
                resized_img.save(buffer, format="PNG", optimize=True, compression_level=9)
            else:
                resized_img.save(buffer, format="JPEG", quality=quality, optimize=True)
            
            buffer.seek(0)
            
            downsampled_size = len(buffer.getvalue()) / 1024  # KB
            
            downsampled_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print(f"Downsampled image: {width}x{height} -> {new_width}x{new_height}, "
                  f"Size: {original_size:.1f}KB -> {downsampled_size:.1f}KB "
                  f"({(downsampled_size/original_size*100):.1f}%) [Mode: {img_mode}]")
            
            return downsampled_b64
    except Exception as e:
        print(f"Error downsampling image {image_path}: {e}")
        return None

def encode_image(image_path, downsample=True, max_size=512):
    """Encode image to base64 with optional downsampling"""
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    try:
        if downsample:
            return downsample_image(image_path, max_size)
        else:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def load_general_content():
    general_content = []
    
    risk_content = """Risk must be calculated linear as a value, value less than 2 is defined as relative low risk,2-5 is moderate risk, greater than 5 is concerned high risk or risky.
Risks added by weather:
Sunny:0
Rainy:+1
Cloudy:+1
Foggy:+2
Night:+2

Risks added by appeared corner cases and Reasons:
Entrance Locating:+1
-Challenge of finding specific small target without visual reference
-risk of missing entrances and become disoriented

DeadEnd:+1
-generally low physical risk

Narrow Path:+2
-limited space causes higher collision risk

Barriers:+3
-unexpected obstacles cause risk of falling, hitting and injury
-challenge of finding a new path

Cross the Road:+3
-multiple moving hazards from different locations
-risk of serious injury

High obstacles:+3
-risk of serious injury in head and face, affecting assistive devices

Uneven Road:+4
-high risk of falling, injury with unexpected elevation changes and road condition changes

Cross the Road in danger:+5
-extremely high risk of serious injury
-no regulated traffic flows, unpredictable moving hazards
-require fast judgement and fast reaction
"""
    general_content.append(risk_content)

    landmarks_content = """landmarks number are presented as:
1-Traffic Lights
2-Bus Stop
3-Entrance or Exit
4-Stairs
5-Square
6-Pedestrain Cross
7-Garbage bin
8-Big Dumpster
9-Gate
10-Bench
11-Motocycle/Biciyle
12-Poles
13-Postbox
14-Map Board
15-High voltage box
16-manhole cover
17-roadside stall
18-Money ATM
    """
    general_content.append(landmarks_content)
    
    return general_content

def generate_qa_pairs(json_path, depth_scale=0.05, min_pixels=100):
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    with open(json_path, "r") as f:
        json_data = json.load(f)
        json_content = json.dumps(json_data, indent=2)
    
    json_dir = os.path.dirname(os.path.abspath(json_path))
    parent_dir = os.path.dirname(json_dir)
    trajectory_name = json_data.get("trajectory", "")
    traj_dir = os.path.join(parent_dir, trajectory_name) if trajectory_name else None
    
    base_dirs = [json_dir, parent_dir]
    if traj_dir:
        base_dirs.append(traj_dir)
    
    print("Generating semantic and depth descriptions...")
    descriptions = convert_json_to_descriptions(
        json_path, 
        base_dirs=base_dirs, 
        depth_scale=depth_scale,
        min_pixels=min_pixels
    )
    
    formatted_descriptions = format_semantic_depth_descriptions(descriptions)
    print(f"Generated descriptions for all views")
    
    rgb_image_paths = []
    rgb_image_view_map = {}
    
    print(f"Extracting RGB image paths:")
    for view in ["walker", "drone", "dog"]:
        if view in json_data.get("views", {}) and "rgb" in json_data["views"][view]:
            rel_path = json_data["views"][view]["rgb"]
            print(f"Found RGB path in JSON: {rel_path} ({view})")
            
            rgb_path = resolve_path(rel_path, base_dirs)
            
            if rgb_path and os.path.exists(rgb_path):
                rgb_image_paths.append(str(rgb_path))
                rgb_image_view_map[view] = str(rgb_path)
                print(f"  ✓ Found RGB image: {rgb_path}")
            else:
                print(f"  ✗ RGB image not found: {rel_path}")
    
    print(f"Found {len(rgb_image_paths)} RGB images out of 3 expected")
    
    if len(rgb_image_paths) == 0:
        print(f"ERROR: No RGB images found. Skipping.")
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None
    
    general_content = load_general_content()
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Risk Assessment Framework:\n{general_content[0]}"},
                {"type": "text", "text": f"Landmarks Information:\n{general_content[1]}"},
                {"type": "text", "text": f"\n{INSTRUCTION_CONTENT}"},
                {"type": "text", "text": f"Frame Json data:\n{json_content}"},
                {"type": "text", "text": formatted_descriptions},
                {"type": "text", "text": IMAGE_DESCRIPTION}
            ]
        }
    ]
    

    valid_image_count = 0
    for img_path in rgb_image_paths:
        encoded_image = encode_image(img_path, downsample=True, max_size=512)
        if encoded_image:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                    "detail": "high"
                }
            })
            valid_image_count += 1
    
    print(f"Added {valid_image_count} valid RGB images to the API request")
    

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4000,
            )
            response_text = response.choices[0].message.content
            
            # calculate #TOKENS to check the cost $$$
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }


            try:
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    qa_pairs = json.loads(json_str)
                    source_info = {
                        "trajectory": json_data.get("trajectory", "unknown"),
                        "frame_id": json_data.get("frame", "unknown").split('/')[0],
                        "frame_position": json_data.get("frame", "unknown")
                    }

                    qa_pairs = {
                        "source": source_info,
                        "qa_pairs": qa_pairs
                    }
                    desc_text = formatted_descriptions
                    
                    return qa_pairs, usage, {"rgb_paths": rgb_image_view_map, "descriptions": desc_text}
                else:
                    print("No JSON array found in the response")
                    print(f"Response: {response_text}")
                    return [], usage, None
            except json.JSONDecodeError as e:
                print(f"Error parsing response as JSON: {e}")
                print(f"Response: {response_text}")
                return [], usage, None
        
        except Exception as e:
            print(f"Error calling OpenAI API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None


def process_trajectory(trajectory_path, test_mode=False, depth_scale=0.05, min_pixels=100, extract_frames=False, qa_root_dir=None, max_workers=None):
    """
    Process a single trajectory directory
    """
    import concurrent.futures
    
    trajectory_dir = Path(trajectory_path)
    trajectory_name = trajectory_dir.name
    
    json_dir = trajectory_dir / "Json"
    if not json_dir.exists() or not json_dir.is_dir():
        print(f"Error: Json directory not found in {trajectory_path}")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, []
    
    # Get all JSON files
    json_files = sorted([f for f in json_dir.glob("*.json")])
    if not json_files:
        print(f"Error: No JSON files found in {json_dir}")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, []
    
    # Skip the first 5 frames and take 1/10 of the total (or just 1 if test_mode is True)
    json_files = json_files[5:]
    if test_mode:
        sample_count = 1
        print("TEST MODE: Processing only one frame")
    else:
        sample_count = max(1, math.ceil(len(json_files) / 10))
    
    sampled_files = random.sample(json_files, sample_count)
    
    print(f"Processing trajectory: {trajectory_name}")
    print(f"Total JSON files (excluding first 5): {len(json_files)}")
    print(f"Sampling {sample_count} files for QA pair generation")
    
    # Create output directory
    output_dir = Path("E:\Dataset\QAPairs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics counters
    stats = defaultdict(int)
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    extracted_frames = []
    
    if not test_mode and max_workers is not None and max_workers > 1 and len(sampled_files) > 1:
        print(f"Using {max_workers} workers to process {len(sampled_files)} frames in parallel")
        
        def process_frame(json_file):
            """处理单个帧的函数"""
            frame_id = json_file.stem
            print(f"Processing frame {frame_id}")
            
            qa_pairs, usage, frame_resources = generate_qa_pairs(
                json_file, 
                depth_scale=depth_scale,
                min_pixels=min_pixels
            )
            
            # Log token usage for this request
            print(f"Token usage for frame {frame_id}: {usage['total_tokens']} total tokens "
                 f"({usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion)")
            
            return {
                "frame_id": frame_id,
                "qa_pairs": qa_pairs,
                "usage": usage,
                "resources": frame_resources
            }
        
        frame_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_frame, json_file): json_file.stem 
                     for json_file in sampled_files}
            
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                try:
                    result = future.result()
                    frame_results.append(result)
                    
                    usage = result["usage"]
                    token_usage["prompt_tokens"] += usage["prompt_tokens"]
                    token_usage["completion_tokens"] += usage["completion_tokens"]
                    token_usage["total_tokens"] += usage["total_tokens"]
                    
                    if result["qa_pairs"]:
                        qa_pairs = result["qa_pairs"]
                        for qa in qa_pairs["qa_pairs"]:
                            if "difficulty" in qa and "category" in qa:
                                stats[f"{qa['difficulty']}-{qa['category']}"] += 1
                        
                        output_file = output_dir / f"{trajectory_name}_{frame_id}_qa.json"
                        with open(output_file, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"Generated {len(qa_pairs['qa_pairs'])} QA pairs for frame {frame_id}")
                    else:
                        print(f"Failed to generate QA pairs for frame {frame_id}")
                        
                except Exception as e:
                    print(f"Error processing frame {frame_id}: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        for json_file in tqdm(sampled_files, desc=f"Generating QA pairs for {trajectory_name}"):
            frame_id = json_file.stem
            
            print(f"\nProcessing frame {frame_id}")
            
            qa_pairs, usage, frame_resources = generate_qa_pairs(
                json_file, 
                depth_scale=depth_scale,
                min_pixels=min_pixels
            )
            
            token_usage["prompt_tokens"] += usage["prompt_tokens"]
            token_usage["completion_tokens"] += usage["completion_tokens"]
            token_usage["total_tokens"] += usage["total_tokens"]
            
            print(f"Token usage for frame {frame_id}: {usage['total_tokens']} total tokens "
                 f"({usage['prompt_tokens']} prompt, {usage['completion_tokens']} completion)")
            
            if qa_pairs:
                for qa in qa_pairs["qa_pairs"]:
                    if "difficulty" in qa and "category" in qa:
                        stats[f"{qa['difficulty']}-{qa['category']}"] += 1
                
                output_file = output_dir / f"{trajectory_name}_{frame_id}_qa.json"
                with open(output_file, "w") as f:
                    json.dump(qa_pairs, f, indent=2)
                print(f"Generated {len(qa_pairs['qa_pairs'])} QA pairs for frame {frame_id}")
            else:
                print(f"Failed to generate QA pairs for frame {frame_id}")
    
    # Print statistics
    print("\nQA Pair Generation Statistics:")
    print("------------------------------")
    print(f"Total frames processed: {len(sampled_files)}")
    
    # Group by difficulty
    easy_count = sum(stats[k] for k in stats if k.startswith("Easy"))
    middle_count = sum(stats[k] for k in stats if k.startswith("Middle"))
    hard_count = sum(stats[k] for k in stats if k.startswith("Hard"))
    
    print(f"Easy level QA pairs: {easy_count}")
    for k in sorted(stats.keys()):
        if k.startswith("Easy"):
            print(f"  - {k}: {stats[k]}")
    
    print(f"Middle level QA pairs: {middle_count}")
    for k in sorted(stats.keys()):
        if k.startswith("Middle"):
            print(f"  - {k}: {stats[k]}")
    
    print(f"Hard level QA pairs: {hard_count}")
    for k in sorted(stats.keys()):
        if k.startswith("Hard"):
            print(f"  - {k}: {stats[k]}")
    
    print(f"Total QA pairs generated: {sum(stats.values())}")
    
    # Print token usage statistics
    print("\nToken Usage Statistics:")
    print("----------------------")
    print(f"Prompt tokens: {token_usage['prompt_tokens']}")
    print(f"Completion tokens: {token_usage['completion_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")
    
    # Save statistics to file
    stats_file = output_dir / f"{trajectory_name}_statistics.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_frames": len(sampled_files),
            "total_qa_pairs": sum(stats.values()),
            "easy_count": easy_count,
            "middle_count": middle_count,
            "hard_count": hard_count,
            "detailed_stats": {k: stats[k] for k in sorted(stats.keys())},
            "token_usage": token_usage,
            "extracted_frames": len(extracted_frames)
        }, f, indent=2)
    
    print(f"Statistics saved to {stats_file}")
    print(f"Extracted {len(extracted_frames)} QA frames")
    
    return token_usage, extracted_frames

# this function will be excecuted if you choose to process multiple trajectories at a time
# the function is no other than process_trajectory
def process_multiple_trajectories(folder_path, test_mode=False, depth_scale=0.05, min_pixels=100, extract_frames=False, qa_root_dir=None):
    print(f"Processing all trajectories in folder: {folder_path}")
    folder_dir = Path(folder_path)

    subdirs = [d for d in folder_dir.iterdir() if d.is_dir()]
    
    trajectory_dirs = [d for d in subdirs if (d / "Json").exists() and (d / "Json").is_dir()]
    
    if not trajectory_dirs:
        print(f"Error: No valid trajectory directories found in {folder_path}")
        return
    
    print(f"Found {len(trajectory_dirs)} trajectory directories")
    
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    all_extracted_frames = []

    for traj_dir in trajectory_dirs:
        print(f"\n{'='*50}")
        print(f"Processing trajectory: {traj_dir.name}")
        print(f"{'='*50}")
        
        try:
            token_usage, extracted_frames = process_trajectory(
                traj_dir, 
                test_mode=test_mode, 
                depth_scale=depth_scale,
                min_pixels=min_pixels,
                extract_frames=extract_frames,
                qa_root_dir=qa_root_dir
            )
            
            total_token_usage["prompt_tokens"] += token_usage["prompt_tokens"]
            total_token_usage["completion_tokens"] += token_usage["completion_tokens"]
            total_token_usage["total_tokens"] += token_usage["total_tokens"]
            
            if extracted_frames:
                all_extracted_frames.extend(extracted_frames)
                
            print(f"Finished processing {traj_dir.name}")
        except Exception as e:
            print(f"Error processing trajectory {traj_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    cost_file = folder_dir / "cost.txt"
    with open(cost_file, "w") as f:
        f.write("GPT-4o Token Usage Summary\n")
        f.write("========================\n\n")
        f.write(f"Prompt tokens: {total_token_usage['prompt_tokens']}\n")
        f.write(f"Completion tokens: {total_token_usage['completion_tokens']}\n")
        f.write(f"Total tokens: {total_token_usage['total_tokens']}\n\n")
        
        prompt_cost = total_token_usage['prompt_tokens'] * 0.0025 / 1000  # $2.50 per 1M tokens
        completion_cost = total_token_usage['completion_tokens'] * 0.01 / 1000  # $10.00 per 1M tokens
        total_cost = prompt_cost + completion_cost
        
        f.write("Estimated Cost (USD)\n")
        f.write("-------------------\n")
        f.write(f"Prompt cost: ${prompt_cost:.2f}\n")
        f.write(f"Completion cost: ${completion_cost:.2f}\n")
        f.write(f"Total cost: ${total_cost:.2f}\n")
    
    print(f"\nTotal token usage across all trajectories:")
    print(f"Prompt tokens: {total_token_usage['prompt_tokens']}")
    print(f"Completion tokens: {total_token_usage['completion_tokens']}")
    print(f"Total tokens: {total_token_usage['total_tokens']}")
    print(f"Token usage summary saved to {cost_file}")

# This function will be executed if you try to generate qa pairs for all trajectories of the same scenario, for example Busstop
# The function is no other than process_trajectory
def process_prefix_trajectories(parent_dir, prefix, **kwargs):
    print(f"Processing all '{prefix}*' trajectories in {parent_dir}")
    parent_path = Path(parent_dir)
    
    trajectory_dirs = [d for d in parent_path.iterdir() 
                      if d.is_dir() and d.name.startswith(prefix) and (d / "Json").exists()]
    
    if not trajectory_dirs:
        print(f"Error: No trajectory directories starting with '{prefix}' found in {parent_dir}")
        return
        trajectory_dirs.sort()
    
    print(f"Found {len(trajectory_dirs)} trajectories with prefix '{prefix}':")
    for traj in trajectory_dirs:
        print(f"  - {traj.name}")
    
    import concurrent.futures
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_extracted_frames = []
    
    max_workers = kwargs.get('max_workers', None)
    if max_workers is None or max_workers <= 0:
        max_workers = min(12, len(trajectory_dirs))
    
    print(f"Starting processing with {max_workers} workers")
    
    def process_single_trajectory(traj_dir):
        traj_name = traj_dir.name
        print(f"Starting processing of trajectory: {traj_name}")
        try:
            token_usage, extracted_frames = process_trajectory(
                traj_dir,
                **{k: v for k, v in kwargs.items() if k != 'max_workers'}
            )
            return {
                "name": traj_name,
                "token_usage": token_usage,
                "extracted_frames": extracted_frames,
                "success": True
            }
        except Exception as e:
            print(f"Error processing trajectory {traj_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "name": traj_name,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "extracted_frames": [],
                "success": False,
                "error": str(e)
            }
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_traj = {executor.submit(process_single_trajectory, traj_dir): traj_dir.name 
                         for traj_dir in trajectory_dirs}
        
        from tqdm import tqdm
        for future in tqdm(concurrent.futures.as_completed(future_to_traj), 
                          total=len(future_to_traj),
                          desc=f"Processing {prefix}* trajectories"):
            traj_name = future_to_traj[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    usage = result["token_usage"]
                    total_token_usage["prompt_tokens"] += usage["prompt_tokens"]
                    total_token_usage["completion_tokens"] += usage["completion_tokens"]
                    total_token_usage["total_tokens"] += usage["total_tokens"]
                    
                    all_extracted_frames.extend(result["extracted_frames"])
                    
                    print(f"Completed processing of trajectory: {traj_name}")
                else:
                    print(f"Failed to process trajectory: {traj_name}")
            except Exception as e:
                print(f"Error getting result for {traj_name}: {e}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\nProcessing Summary for '{prefix}*' trajectories:")
    print(f"Successfully processed {success_count} of {len(trajectory_dirs)} trajectories")
    
    print("\nToken Usage Statistics:")
    print("----------------------")
    print(f"Prompt tokens: {total_token_usage['prompt_tokens']}")
    print(f"Completion tokens: {total_token_usage['completion_tokens']}")
    print(f"Total tokens: {total_token_usage['total_tokens']}")
    
    prompt_cost = total_token_usage['prompt_tokens'] * 0.0025 / 1000
    completion_cost = total_token_usage['completion_tokens'] * 0.01 / 1000
    total_cost = prompt_cost + completion_cost
    
    print("\nEstimated Cost (USD):")
    print("-------------------")
    print(f"Prompt cost: ${prompt_cost:.2f}")
    print(f"Completion cost: ${completion_cost:.2f}")
    print(f"Total cost: ${total_cost:.2f}")
    
    stats_file = Path(parent_dir) / f"{prefix}_trajectories_statistics.json"
    with open(stats_file, "w") as f:
        json.dump({
            "prefix": prefix,
            "total_trajectories": len(trajectory_dirs),
            "successful_trajectories": success_count,
            "token_usage": total_token_usage,
            "estimated_cost": {
                "prompt_cost": prompt_cost,
                "completion_cost": completion_cost,
                "total_cost": total_cost
            }
        }, f, indent=2)
    
    print(f"Statistics saved to {stats_file}")
    
    return total_token_usage, all_extracted_frames

def main():
    print("Starting qaGen.py main function...")
    parser = argparse.ArgumentParser(description='Generate QA pairs for trajectory data using GPT-4o')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-path', type=str,
                       help='Path to a single trajectory directory or a parent directory for prefix mode')
    group.add_argument('-pathall', type=str,
                       help='Path to a folder containing multiple trajectory directories')
    
    parser.add_argument('-prefix', type=str, default=None,
                       help='Process all trajectories with this prefix when using -path with a parent directory')
    
    parser.add_argument('-test', action='store_true',
                        help='Test mode: only process one frame per trajectory')
    parser.add_argument('-api_key', type=str,
                        help='OpenAI API key (overrides the one in the script)')
    parser.add_argument('-depth_scale', type=float, default=0.05,
                        help='Scale factor to convert depth values to meters (default: 0.05)')
    parser.add_argument('-min_pixels', type=int, default=100,
                        help='Minimum number of pixels required to consider an object (default: 100)')
    parser.add_argument('-extract', action='store_true',
                        help='Extract QA frames to separate directory')
    parser.add_argument('-qa_dir', type=str,
                        help='Directory to extract QA frames (default: [dataset_root]/QA)')
    parser.add_argument('-workers', type=int, default=None,
                        help='Number of parallel workers for processing frames or trajectories')
    
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")
    
    if args.path:
        target_path = args.path
        process_all = False
    else:
        target_path = args.pathall
        process_all = True
    
    test_mode = args.test
    depth_scale = args.depth_scale
    min_pixels = args.min_pixels
    extract_frames = args.extract
    
    # Check if path exists
    if not os.path.exists(target_path):
        print(f"Error: Path {target_path} does not exist")
        return
    
    # Set API key if provided as argument
    global OPENAI_API_KEY
    if args.api_key:
        OPENAI_API_KEY = args.api_key
    
    # Check for OpenAI API key
    if OPENAI_API_KEY == "api":
        print("Error: Please set your OpenAI API key in the script or provide it with -api_key")
        return
    
    qa_root_dir = None
    if extract_frames:
        if args.qa_dir:
            qa_root_dir = args.qa_dir
        else:
            dataset_root = os.path.abspath(target_path)
            if process_all:
                qa_root_dir = os.path.join(dataset_root, "QA")
            else:
                if os.path.isdir(dataset_root) and (Path(dataset_root) / "Json").exists():
                    dataset_root = os.path.dirname(dataset_root)
                qa_root_dir = os.path.join(dataset_root, "QA")
        
        print(f"QA frames will be extracted to: {qa_root_dir}")
        os.makedirs(qa_root_dir, exist_ok=True)
    
    try:
        if process_all:
            process_multiple_trajectories(
                target_path,
                test_mode=test_mode,
                depth_scale=depth_scale,
                min_pixels=min_pixels,
                extract_frames=extract_frames,
                qa_root_dir=qa_root_dir,
                max_workers=args.workers
            )
        else:
            if args.prefix:
                if os.path.isdir(target_path) and not os.path.isdir(os.path.join(target_path, "Json")):
                    print(f"Processing all trajectories with prefix '{args.prefix}' in {target_path}")
                    process_prefix_trajectories(
                        target_path,
                        args.prefix,
                        test_mode=test_mode,
                        depth_scale=depth_scale,
                        min_pixels=min_pixels,
                        extract_frames=extract_frames,
                        qa_root_dir=qa_root_dir,
                        max_workers=args.workers
                    )
                else:
                    print(f"Error: When using -prefix, path should be a parent directory containing multiple trajectories")
                    return
            else:
                print(f"Processing single trajectory: {target_path}")
                process_trajectory(
                    target_path, 
                    test_mode=test_mode, 
                    depth_scale=depth_scale,
                    min_pixels=min_pixels,
                    extract_frames=extract_frames,
                    qa_root_dir=qa_root_dir,
                    max_workers=args.workers
                )
        
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error in processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Executing qaGen.py script")
    main()