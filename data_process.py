import os
import argparse
import json
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import threading

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):

    scenario_types = None

    scenario_tokens                      # List of scenario tokens to include
    log_names = log_names                # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type               # Number of scenarios per type
    limit_total_scenarios                # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None         # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = True              # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = False          # Whether to remove scenarios where the mission goal is invalid
    shuffle                              # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

def process_scenario_batch(scenarios_batch, args, batch_id):
    """
    处理一批场景的函数，用于多进程执行
    """
    processor = DataProcessor(args)
    results = []
    
    # 为每个批次创建进度条
    batch_pbar = tqdm(scenarios_batch, 
                     desc=f"批次 {batch_id}", 
                     position=batch_id,
                     leave=False,
                     ncols=100)
    
    for scenario in batch_pbar:
        try:
            # 单独处理每个场景
            map_name = scenario._map_name
            token = scenario.token
            result = processor.process_single_scenario(scenario)
            if result is not None:
                processor.save_to_disk(args.save_path, result)
                results.append(f"{map_name}_{token}.npz")
                
            # 更新进度条描述
            batch_pbar.set_postfix({
                '已完成': len(results),
                '成功率': f"{len(results)/len(scenarios_batch)*100:.1f}%"
            })
            
        except Exception as e:
            print(f"处理场景时出错 {scenario.token}: {str(e)}")
            continue
    
    batch_pbar.close()
    return results, batch_id

def split_scenarios_into_batches(scenarios, num_workers):
    """
    将场景列表分割成批次供多进程处理
    """
    batch_size = max(1, len(scenarios) // num_workers)
    batches = []
    
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        batches.append(batch)
    
    return batches

def print_progress_summary(completed_batches, total_batches, total_scenarios, start_time):
    """
    打印进度摘要
    """
    elapsed_time = time.time() - start_time
    progress_pct = completed_batches / total_batches * 100
    
    print(f"\n{'='*60}")
    print(f"进度摘要:")
    print(f"  已完成批次: {completed_batches}/{total_batches} ({progress_pct:.1f}%)")
    print(f"  已用时间: {elapsed_time:.1f}秒")
    
    if completed_batches > 0:
        avg_time_per_batch = elapsed_time / completed_batches
        estimated_total_time = avg_time_per_batch * total_batches
        remaining_time = estimated_total_time - elapsed_time
        print(f"  预计剩余时间: {remaining_time:.1f}秒")
        print(f"  预计总时间: {estimated_total_time:.1f}秒")
    
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')

    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')

    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # 新增多进程相关参数
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count()//2, 
                       help='number of worker processes for parallel processing')
    parser.add_argument('--use_multiprocessing', action='store_true', 
                       help='enable multiprocessing for faster data processing')
    parser.add_argument('--show_detailed_progress', action='store_true', default=True,
                       help='show detailed progress information')
    
    args = parser.parse_args()

    print(f"🚀 开始数据处理...")
    print(f"📁 数据路径: {args.data_path}")
    print(f"🗺️  地图路径: {args.map_path}")
    print(f"💾 保存路径: {args.save_path}")
    print(f"🎯 目标场景数: {args.total_scenarios}")
    
    if args.use_multiprocessing:
        print(f"⚡ 多进程模式: 启用 ({args.num_workers} 个进程)")
    else:
        print(f"🐌 单进程模式: 启用")
    print(f"{'='*60}")

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    sensor_root = None
    db_files = None

    # Only preprocess the training data
    print("📖 加载训练数据配置...")
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)

    print("🏗️  构建场景...")
    map_version = "nuplan-maps-v1.0"    
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"✅ 场景加载完成，总数: {len(scenarios)}")

    # process data
    del worker, builder, scenario_filter
    
    start_time = time.time()
    
    if args.use_multiprocessing and len(scenarios) > 1:
        print(f"\n⚡ 启动多进程处理 (进程数: {args.num_workers})")
        
        # 将场景分割成批次
        scenario_batches = split_scenarios_into_batches(scenarios, args.num_workers)
        print(f"📦 场景分为 {len(scenario_batches)} 个批次")
        
        # 显示批次信息
        for i, batch in enumerate(scenario_batches):
            print(f"   批次 {i+1}: {len(batch)} 个场景")
        print()
        
        all_npz_files = []
        completed_batches = 0
        
        # 创建总进度条
        total_pbar = tqdm(total=len(scenarios), 
                         desc="总进度", 
                         position=len(scenario_batches),
                         ncols=100,
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # 使用进程池执行器
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有批次任务
            futures = {}
            for i, batch in enumerate(scenario_batches):
                future = executor.submit(process_scenario_batch, batch, args, i+1)
                futures[future] = i+1
                
            print(f"📤 已提交 {len(futures)} 个批次任务\n")
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    batch_results, batch_id = future.result()
                    all_npz_files.extend(batch_results)
                    completed_batches += 1
                    
                    # 更新总进度条
                    total_pbar.update(len(batch_results))
                    total_pbar.set_postfix({
                        '批次': f"{completed_batches}/{len(scenario_batches)}",
                        '成功': len(all_npz_files),
                        '成功率': f"{len(all_npz_files)/len(scenarios)*100:.1f}%"
                    })
                    
                    # 显示详细进度
                    if args.show_detailed_progress:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / len(all_npz_files) if all_npz_files else 0
                        print(f"\n✅ 批次 {batch_id} 完成: {len(batch_results)} 个场景")
                        print(f"   累计完成: {len(all_npz_files)}/{len(scenarios)} ({len(all_npz_files)/len(scenarios)*100:.1f}%)")
                        print(f"   平均耗时: {avg_time:.3f}秒/场景")
                        
                except Exception as e:
                    print(f"❌ 批次处理失败: {str(e)}")
        
        total_pbar.close()
        npz_files = all_npz_files
        
    else:
        print("\n🐌 使用单进程处理")
        processor = DataProcessor(args)
        
        # 单进程也显示进度条
        scenarios_pbar = tqdm(scenarios, desc="处理场景", ncols=100)
        processed_files = []
        
        for scenario in scenarios_pbar:
            result = processor.process_single_scenario(scenario)
            if result is not None:
                processor.save_to_disk(args.save_path, result)
                processed_files.append(f"{result['map_name']}_{result['token']}.npz")
            
            scenarios_pbar.set_postfix({
                '已完成': len(processed_files),
                '成功率': f"{len(processed_files)/(scenarios_pbar.n+1)*100:.1f}%"
            })
        
        npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    end_time = time.time()
    processing_time = end_time - start_time
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"🎉 数据处理完成!")
    print(f"⏱️  总耗时: {processing_time:.2f} 秒")
    print(f"📊 处理统计:")
    print(f"   - 目标场景数: {len(scenarios)}")
    print(f"   - 成功处理: {len(npz_files)}")
    print(f"   - 成功率: {len(npz_files)/len(scenarios)*100:.1f}%")
    print(f"   - 平均耗时: {processing_time/len(scenarios):.3f} 秒/场景")
    
    if args.use_multiprocessing:
        speedup = len(scenarios) / processing_time if processing_time > 0 else 0
        theoretical_speedup = args.num_workers
        efficiency = speedup / theoretical_speedup * 100 if theoretical_speedup > 0 else 0
        print(f"⚡ 并行效率:")
        print(f"   - 使用进程数: {args.num_workers}")
        print(f"   - 实际加速比: {speedup:.1f}x")
        print(f"   - 并行效率: {efficiency:.1f}%")
    
    print(f"{'='*60}")

    # Save the list to a JSON file
    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"💾 已保存 {len(npz_files)} 个文件名到 diffusion_planner_training.json")