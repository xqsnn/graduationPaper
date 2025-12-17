"""
测试NSGA2求解器保存功能的脚本
"""
from algorithm.multiple_objective.nsga2_solver import NSGA2Solver, create_sample_materials, StaticParameters
from algorithm.multiple_objective.database_util import save_pareto_front_results
import uuid
from datetime import datetime

def test_save_function():
    """测试保存功能"""
    print("创建示例材料数据...")
    materials = create_sample_materials()[:5]  # 只使用前5个材料以加快测试
    print(f"成功创建 {len(materials)} 个示例材料")
    
    print("\n创建求解器...")
    params = StaticParameters()
    solver = NSGA2Solver(materials, params)
    
    print("\n开始求解（小规模测试）...")
    # 使用小规模参数进行快速测试
    pareto_front = solver.solve(
        pop_size=20,      # 减小种群大小
        n_generations=10, # 减小迭代次数
        mutation_rate=0.15,
        verbose=True
    )
    
    print(f"\n获得 {len(pareto_front)} 个帕累托最优解")
    
    print("\n测试保存到数据库...")
    task_id = f"test_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    save_pareto_front_results(task_id, pareto_front, materials)
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_save_function()