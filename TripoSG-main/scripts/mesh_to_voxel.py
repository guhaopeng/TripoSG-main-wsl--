import trimesh
import numpy as np
import os
import argparse
from pathlib import Path

def mesh_to_voxel(mesh_path, voxel_resolution=32, output_path=None):
    """
    将3D网格模型转换为体素表示
    
    Args:
        mesh_path: GLB/OBJ文件路径
        voxel_resolution: 体素化分辨率
        output_path: 输出文件路径(.npy格式)
    
    Returns:
        voxels: 体素网格数组
    """
    print(f"正在加载网格文件: {mesh_path}")
    mesh = trimesh.load(mesh_path)
    
    if isinstance(mesh, trimesh.Scene):
        # 如果加载的是场景，提取第一个网格
        geometries = list(mesh.geometry.values())
        if len(geometries) > 0:
            mesh = geometries[0]
    
    print(f"网格信息:")
    print(f"- 顶点数: {len(mesh.vertices)}")
    print(f"- 面片数: {len(mesh.faces)}")
    
    # 将网格转换为体素表示
    print(f"\n开始体素化(分辨率: {voxel_resolution}x{voxel_resolution}x{voxel_resolution})...")
    voxels = mesh.voxelized(pitch=1.0/voxel_resolution)
    
    # 获取密集的体素网格表示
    dense_voxels = voxels.matrix
    print(f"体素化完成，形状: {dense_voxels.shape}")
    print(f"非零体素数量: {np.count_nonzero(dense_voxels)}")
    
    if output_path:
        # 保存体素数据
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        np.save(output_path, dense_voxels)
        print(f"\n体素数据已保存至: {output_path}")
    
    return dense_voxels

def main():
    parser = argparse.ArgumentParser(description="将3D网格模型转换为体素表示")
    parser.add_argument("--mesh-path", type=str, required=True, help="输入的3D网格模型文件路径(.glb/.obj)")
    parser.add_argument("--voxel-resolution", type=int, default=32, help="体素化分辨率")
    parser.add_argument("--output-path", type=str, default=None, help="输出文件路径(.npy)")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，则根据输入文件名生成
    if args.output_path is None:
        input_path = Path(args.mesh_path)
        args.output_path = str(input_path.parent / f"{input_path.stem}_voxels.npy")
    
    try:
        voxels = mesh_to_voxel(
            args.mesh_path,
            args.voxel_resolution,
            args.output_path
        )
        print("\n转换成功完成!")
        
    except Exception as e:
        print(f"\n转换过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()