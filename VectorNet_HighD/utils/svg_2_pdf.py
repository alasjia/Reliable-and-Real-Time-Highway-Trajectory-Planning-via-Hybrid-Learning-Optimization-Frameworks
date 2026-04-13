import cairosvg
from pathlib import Path

def batch_convert_svg_to_pdf(input_folder, output_folder):
    # 将字符串路径转换为 Path 对象
    src_dir = Path(input_folder)
    dest_dir = Path(output_folder)

    # 如果输出目录不存在，自动创建
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有 .svg 文件
    svg_files = list(src_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"? 在 {src_dir} 中没找到任何 SVG 文件。")
        return

    print(f"🚀 开始转换任务，共 {len(svg_files)} 个文件...")

    for svg_path in svg_files:
        # 构造输出文件名 (将 .svg 替换为 .pdf)
        pdf_path = dest_dir / f"{svg_path.stem}.pdf"
        
        try:
            # 执行转换
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
            print(f"✅ 已转换: {svg_path.name} -> {pdf_path.name}")
        except Exception as e:
            print(f"❌ 转换失败 {svg_path.name}: {e}")

    print("\n✨ 所有任务已完成！")

# --- 使用示例 ---
# 替换为你的文件夹路径
input_dir = r'E:\UTC_PHD\Compared1_VN_trajectory_prediction_chwei\replanning\paper_pics_generation\Results\displaying_pics\pre与gt横向上差距较大的BEVfigs'
output_dir = r'E:\UTC_PHD\Compared1_VN_trajectory_prediction_chwei\replanning\paper_pics_generation\Results\displaying_pics\pre与gt横向上差距较大的BEVfigs'

batch_convert_svg_to_pdf(input_dir, output_dir)