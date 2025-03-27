import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages  
import os  
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

def pro_block_value_diff():
    # 读取log文件
    with open('/home/lianyaoxiu/lianyaoxiu/Infinity/block.log', 'r') as file:
        log_data = file.readlines()

    # 初始化数据结构
    scale_data = {}

    # 解析log文件
    for i, line in enumerate(log_data):
        if re.match(r'Scale \d+, Block \d+ Difference', line):
            next_line = log_data[i + 1]
            values = re.findall(r'<(\d+\.?\d*): (\d+\.?\d*)%', next_line)
            scale = re.search(r'Scale (\d+)', line).group(1)
            block = re.search(r'Block (\d+)', line).group(1)
            
            if scale not in scale_data:
                scale_data[scale] = {}
            scale_data[scale][block] = {k: float(v) for k, v in values}

    # 定义颜色
    colors = {
        '<10%': 'blue',
        '<5%': 'pink',
        '<1%': 'green',
        '<0.1%': 'red',
        '<0.05%': 'orange',
        '<0.01%': 'purple'
    }

    # 绘制折线图
    for scale, blocks in scale_data.items():
        blocks_sorted = sorted(blocks.items(), key=lambda x: int(x[0]))
        x = range(len(blocks_sorted))
        
        # 提取<10, <1, <0.1的数据
        y_10 = [block[1].get('10', 0) for block in blocks_sorted]
        y_5 = [block[1].get('5', 0) for block in blocks_sorted]
        y_1 = [block[1].get('1', 0) for block in blocks_sorted]
        
        # 提取<0.1, <0.05, <0.01的数据
        y_01_right = [block[1].get('0.1', 0) for block in blocks_sorted]
        y_005 = [block[1].get('0.05', 0) for block in blocks_sorted]
        y_001 = [block[1].get('0.01', 0) for block in blocks_sorted]
        
        # 创建画布
        plt.figure(figsize=(12, 6))
        
        # 左边图：<10, <5, <1
        plt.subplot(1, 2, 1)
        plt.plot(x, y_10, label='<10%', color=colors['<10%'], marker='o')
        plt.plot(x, y_5, label='<5%', color=colors['<5%'], marker='s')
        plt.plot(x, y_1, label='<1%', color=colors['<1%'], marker='^')
        plt.xlabel('Block Number')
        plt.ylabel('Percentage')
        plt.title(f'Scale {scale}: <10, <5, <1%')
        plt.legend()
        
        # 右边图：<0.1, <0.05, <0.01
        plt.subplot(1, 2, 2)
        plt.plot(x, y_01_right, label='<0.1%', color=colors['<0.1%'], marker='o')
        plt.plot(x, y_005, label='<0.05%', color=colors['<0.05%'], marker='s')
        plt.plot(x, y_001, label='<0.01%', color=colors['<0.01%'], marker='^')
        plt.xlabel('Block Number')
        plt.ylabel('Percentage')
        plt.title(f'Scale {scale}: <0.1, <0.05, <0.01%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'scale_{scale}_plot.png', dpi=300)
        plt.close()  # 关闭当前图，避免内存泄漏

def extract_number(filename):
    """
    从文件名中提取数字部分。
    例如, scale_0_plot.png -> 0, scale_1_plot.png -> 1
    """
    match = re.search(r'\d+', filename)  # 查找文件名中的数字
    if match:
        return int(match.group())  # 返回提取的数字
    return 0  # 如果没有数字，返回默认值

def images_to_pdf():
    """
    将文件夹中的图像拼接在一起，输出为 PDF 文件。

    :param image_folder: 包含图像的文件夹路径
    :param output_pdf: 输出的 PDF 文件路径
    """
    image_folder = "/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/profile"  # 替换为你的图像文件夹路径
    output_pdf = "value_diff.pdf"  # 输出的 PDF 文件路径
    # 获取文件夹中的所有图像文件
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    image_files.sort(key=lambda x: extract_number(os.path.basename(x)))

    if not image_files:
        print("文件夹中没有图像文件！")
        return

    # 创建 PDF 文件
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4  # 获取 A4 纸的尺寸

    for image_file in image_files:
        img = Image.open(image_file)
        img_width, img_height = img.size

        # 计算缩放比例，使图像适应 A4 页面
        scale = min(width / img_width, height / img_height)
        img_width *= scale
        img_height *= scale

        # 居中放置图像
        x = (width - img_width) / 2
        y = (height - img_height) / 2

        # 将图像添加到 PDF 页面
        c.drawImage(image_file, x, y, width=img_width, height=img_height)
        c.showPage()  # 结束当前页面

    c.save()
    print(f"PDF 文件已生成：{output_pdf}")

def pro_block_cos_diff():

    def process_log_file(input_file_path, output_pdf_path):  
        """  
        处理日志文件并生成图表  
        
        Args:  
            input_file_path: 输入的日志文件路径  
            output_pdf_path: 输出的PDF文件路径  
        """  
        # 检查输入文件是否存在  
        if not os.path.exists(input_file_path):  
            raise FileNotFoundError(f"找不到输入文件: {input_file_path}")  
        
        # 确保输出目录存在  
        output_dir = os.path.dirname(output_pdf_path)  
        if output_dir and not os.path.exists(output_dir):  
            os.makedirs(output_dir)  

        # 读取并解析日志文件  
        stages = []  
        current_stage_data = ([], [])  
        
        try:  
            with open(input_file_path, 'r', encoding='utf-8') as file:  
                for line in file:  
                    line = line.strip()  
                    if not line:  # 跳过空行  
                        continue  
                        
                    if line.startswith("矩阵A和B的余弦相似度为:"):  
                        try:  
                            # 分离数值部分  
                            _, values = line.split(":")  
                            value1, value2 = map(float, values.strip().split())  
                            current_stage_data[0].append(value1)  
                            current_stage_data[1].append(value2)  
                        except (ValueError, IndexError) as e:  
                            print(f"警告：无法解析行: {line}")  
                            print(f"错误信息: {str(e)}")  
                            continue  
                            
                    elif line.startswith("stage"):  
                        # 如果当前stage有数据，保存它  
                        if current_stage_data[0] or current_stage_data[1]:  
                            stages.append(current_stage_data)  
                            current_stage_data = ([], [])  
        
            # 添加最后一个stage的数据（如果有的话）  
            if current_stage_data[0] or current_stage_data[1]:  
                stages.append(current_stage_data)  
        
            # 绘制图表并保存为PDF  
            with PdfPages(output_pdf_path) as pdf:  
                for idx, (values1, values2) in enumerate(stages):  
                    # 创建新的图表  
                    plt.figure(figsize=(15, 6))  
                    
                    # 设置整体标题  
                    plt.suptitle(f'Stage {idx} Cosine Similarity', fontsize=16)  
                    
                    # 左侧子图  
                    plt.subplot(1, 2, 1)  
                    plt.plot(values1, 'b-', marker='o', markersize=4, label=f'Column 1')  
                    plt.title('Batch1')  
                    plt.xlabel('Sample Index')  
                    plt.ylabel('Cosine Similarity')  
                    plt.grid(True, linestyle='--', alpha=0.7)  
                    plt.legend()  
                    
                    # 右侧子图  
                    plt.subplot(1, 2, 2)  
                    plt.plot(values2, 'r-', marker='o', markersize=4, label=f'Column 2')  
                    plt.title('Batch2')  
                    plt.xlabel('Sample Index')  
                    plt.ylabel('Cosine Similarity')  
                    plt.grid(True, linestyle='--', alpha=0.7)  
                    plt.legend()  
                    
                    # 调整布局  
                    plt.tight_layout()  
                    
                    # 保存当前图表  
                    pdf.savefig()  
                    plt.close()  
                    
            print(f"成功处理了 {len(stages)} 个stages的数据")  
            print(f"PDF文件已保存至: {output_pdf_path}")  
            
        except Exception as e:  
            print(f"处理文件时发生错误: {str(e)}")  
            raise  

    # 设置输入输出路径  
    input_file = "/home/lianyaoxiu/lianyaoxiu/Infinity/cos.log"  # 替换为实际的输入文件路径  
    output_file = "/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/profile/cos_diff.pdf"  # 替换为实际的输出文件路径  
    
    process_log_file(input_file, output_file)  


def parse_log_file(log_file_path):  
    """  
    解析日志文件，提取每个stage的block差异值  
    
    Args:  
        log_file_path: 日志文件路径  
    Returns:  
        list: 包含每个stage数据的列表  
    """  
    stages = []  
    current_stage = []  
    
    try:  
        with open(log_file_path, 'r', encoding='utf-8') as file:  
            for line in file:  
                line = line.strip()  
                if not line:  # 跳过空行  
                    continue  
                
                # 匹配Scale行  
                match = re.match(r'Scale (\d+), Block (\d+) Difference: ([\d.]+)', line)  
                if match:  
                    scale, block, diff = match.groups()  
                    current_stage.append((int(block), float(diff)))  
                
                # 检测stage行，表示一个stage结束  
                elif line.startswith('stage'):  
                    if current_stage:  
                        stages.append(current_stage)  
                        current_stage = []  
    
        # 添加最后一个stage的数据（如果有的话）  
        if current_stage:  
            stages.append(current_stage)  
            
    except Exception as e:  
        print(f"读取文件时发生错误: {str(e)}")  
        return []  
        
    return stages  


def pro_block_sum_diff():  

    input_file = "/home/lianyaoxiu/lianyaoxiu/Infinity/abssum.log"  # 替换为实际的日志文件路径  
    output_pdf = "/home/lianyaoxiu/lianyaoxiu/Infinity/outputs/profile/sum_diff.pdf"  # 替换为实际的输出PDF路径  

    # 确保输出目录存在  
    output_dir = os.path.dirname(output_pdf)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
    
    # 解析日志文件  
    print("开始解析日志文件...")  
    stages = parse_log_file(input_file)  
    
    # 生成图表  
    print(f"找到 {len(stages)} 个stages，开始生成图表...")   
    with PdfPages(output_pdf) as pdf:  
        for stage_idx, stage_data in enumerate(stages):  
            # 分离block编号和difference值  
            blocks, differences = zip(*stage_data)  
            
            # 创建新图  
            plt.figure(figsize=(12, 6))  
            
            # 绘制折线图  
            plt.plot(blocks, differences, 'bo-', markersize=4,   
                    label=f'Stage {stage_idx}')  
            
            # 设置图表标题和标签  
            plt.title(f'Stage {stage_idx} Sum Differences')  
            plt.xlabel('Block Number')  
            plt.ylabel('Difference')  
            plt.grid(True, linestyle='--', alpha=0.7)  
            plt.legend()  
            
            # 调整布局  
            plt.tight_layout()  
            
            # 保存到PDF  
            pdf.savefig()  
            plt.close()  
            
    print(f"已成功生成PDF文件: {output_pdf}")  
    
      
if __name__ == '__main__':
    # pro_block_value_diff()
    # images_to_pdf()
    # pro_block_cos_diff()
    pro_block_sum_diff()
