import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

def visualize_voxels(voxel_path, output_path=None, alpha=0.3):
    """
    可视化体素数据
    
    Args:
        voxel_path: NPY文件路径
        output_path: 输出图像路径（可选）
        alpha: 体素透明度
    """
    print(f"正在加载体素数据: {voxel_path}")
    voxels = np.load(voxel_path)
    
    print(f"体素数据形状: {voxels.shape}")
    print(f"非零体素数量: {np.count_nonzero(voxels)}")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 直接使用voxels函数绘制体素
    # 设置黄色(类似图片中的颜色)
    yellow_color = np.array([1.0, 1.0, 0.0, alpha])  # RGBA
    
    # 绘制体素
    ax.voxels(voxels, 
              facecolors=yellow_color,
              edgecolor='gray',
              linewidth=0.3)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置轴范围
    ax.set_xlim([0, voxels.shape[0]])
    ax.set_ylim([0, voxels.shape[1]])
    ax.set_zlim([0, voxels.shape[2]])
    
    # 设置标题
    plt.title('3D体素可视化')
    
    # 调整布局
    plt.tight_layout()
    
    if output_path:
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存至: {output_path}")
    
    # 显示图像
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可视化体素数据")
    parser.add_argument("--voxel-path", type=str, required=True, help="输入的体素数据文件路径(.npy)")
    parser.add_argument("--output-path", type=str, default=None, help="输出图像路径(.png/.jpg)")
    parser.add_argument("--alpha", type=float, default=0.3, help="体素透明度(0-1)")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，则根据输入文件名生成
    if args.output_path is None:
        input_path = Path(args.voxel_path)
        args.output_path = str(input_path.parent / f"{input_path.stem}_vis.png")
    
    try:
        visualize_voxels(
            args.voxel_path,
            args.output_path,
            args.alpha
        )
        print("\n可视化完成!")
        
    except Exception as e:
        print(f"\n可视化过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()