'''
python Keye_VL_Caption.py Prince_Ciel_Phantomhive_Sebastian_Michaelis_both_Videos_qwen_vl_captioned Prince_Ciel_Phantomhive_Sebastian_Michaelis_both_Videos_keye_captioned --use_flash_attention \
--text "给你的视频中可能出现的主要人物为两个（可能出现一个或两个），当人物为一个戴眼罩的男孩时，男孩的名字是'夏尔',当人物是一个穿燕尾西服的成年男子时，男子的名字是'塞巴斯蒂安',在你的视频描述中要使用人物的名字并且简单描述人物的外貌及衣着。 使用中文描述这个视频 /think"

python Keye_VL_Caption.py Skirk_Images_Captioned Skirk_Keye_VL_Images_Captioned --use_flash_attention --text "画面中的人物是丝柯克，请结合人物的衣着细节和画面背景给出图片的中文描述,在描述中必须提到人物名称为丝柯克 /no_think"
'''

import os
import torch
import argparse
from pathlib import Path
import shutil
from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info
from moviepy.editor import VideoFileClip
import re
import logging

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        with VideoFileClip(video_path) as video:
            return video.duration
    except Exception as e:
        print(f"获取视频时长失败 {video_path}: {e}")
        return float('inf')

class KeyeVL_Captioner:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_model(self):
        """设置和加载Keye-VL模型"""
        try:
            if self.model is None:
                self.model = AutoModel.from_pretrained(
                    self.args.model_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if self.args.use_flash_attention else "eager",
                ).eval()
                self.model.to(self.device)
            
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(
                    self.args.model_path, 
                    trust_remote_code=True
                )
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        
    def determine_thinking_mode(self, text):
        """根据文本内容确定思考模式"""
        if text.endswith('/no_think'):
            return "no_think", text.replace('/no_think', '').strip()
        elif text.endswith('/think'):
            return "think", text.replace('/think', '').strip()
        else:
            return "auto", text
    
    def process_media(self, media_path, output_dir):
        """处理单个媒体文件（图片或视频）"""
        try:
            # 检查文件是否存在
            if not os.path.exists(media_path):
                print(f"文件不存在: {media_path}")
                return None
            
            # 获取文件扩展名确定媒体类型
            ext = os.path.splitext(media_path)[1].lower()
            is_video = ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
            is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            
            if not (is_video or is_image):
                print(f"不支持的文件格式: {media_path}")
                return None
            
            # 视频时长过滤
            if is_video:
                duration = get_video_duration(media_path)
                print(f"视频: {media_path}, 时长: {duration}秒")
                if self.args.max_duration > 0 and duration > self.args.max_duration:
                    print(f"跳过时长超过限制的视频: {duration}秒 > {self.args.max_duration}秒")
                    return None
            
            # 确定思考模式
            thinking_mode, processed_text = self.determine_thinking_mode(self.args.text)
            processed_text = self.args.text
            
            # 准备媒体输入
            media_content = []
            if is_video:
                media_content.append({
                    "type": "video",
                    "video": media_path,
                    "fps": self.args.fps,
                    "max_frames": self.args.max_frames
                })
            else:
                media_content.append({
                    "type": "image",
                    "image": media_path
                })
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": media_content + [
                        {"type": "text", "text": processed_text },
                    ],
                }
            ]
            
            # 处理视觉信息并生成输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs
            )
            inputs = inputs.to(self.device)
            
            # 生成描述
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.args.max_new_tokens,
                    temperature=self.args.temperature,
                    do_sample=self.args.temperature > 0
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
            
            result = output_text[0] if isinstance(output_text, list) else output_text
            
            # 清理结果
            #result = re.sub(r'<[^>]*>', '', result).strip()

            # 保存结果
            media_name = os.path.basename(media_path)
            txt_filename = os.path.splitext(media_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result)
            
            # 复制媒体文件到输出目录
            output_media_path = os.path.join(output_dir, media_name)
            shutil.copy2(media_path, output_media_path)
            
            print(f"文件: {media_name}")
            print(f"思考模式: {thinking_mode}")
            print(f"描述: {result}")
            print("-" * 50)
            
            return result
            
        except Exception as e:
            print(f"处理文件 {media_path} 时发生异常: {e}")
            logging.error(f"处理文件 {media_path} 时发生异常: {e}")
            return None
    
    def process_all_media(self):
        """处理所有媒体文件"""
        try:
            os.makedirs(self.args.output_dir, exist_ok=True)
            self.setup_model()
            
            # 设置日志记录
            logging.basicConfig(
                filename=os.path.join(self.args.output_dir, 'processing_errors.log'),
                level=logging.ERROR,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
            processed_count = 0
            failed_count = 0
            skipped_count = 0
            
            if os.path.isfile(self.args.source_path):
                # 处理单个文件
                try:
                    result = self.process_media(self.args.source_path, self.args.output_dir)
                    if result is None:
                        skipped_count += 1
                        print(f"跳过文件: {self.args.source_path}")
                    else:
                        processed_count += 1
                except Exception as e:
                    failed_count += 1
                    print(f"处理文件 {self.args.source_path} 时发生异常: {e}")
                    logging.error(f"处理文件 {self.args.source_path} 时发生异常: {e}")
                    
            elif os.path.isdir(self.args.source_path):
                supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', 
                                      '.jpg', '.jpeg', '.png', '.bmp', '.gif']
                
                for file in os.listdir(self.args.source_path):
                    if any(file.lower().endswith(ext) for ext in supported_extensions):
                        media_path = os.path.join(self.args.source_path, file)
                        try:
                            result = self.process_media(media_path, self.args.output_dir)
                            if result is None:
                                skipped_count += 1
                                print(f"跳过文件: {file}")
                            else:
                                processed_count += 1
                        except Exception as e:
                            failed_count += 1
                            print(f"处理文件 {file} 时发生异常: {e}")
                            logging.error(f"处理文件 {file} 时发生异常: {e}")
                            continue
            
            # 输出处理统计信息
            print(f"\n处理完成!")
            print(f"成功处理: {processed_count} 个文件")
            print(f"跳过: {skipped_count} 个文件")
            print(f"失败: {failed_count} 个文件")
            print(f"详细错误信息请查看: {os.path.join(self.args.output_dir, 'processing_errors.log')}")
            
        except Exception as e:
            print(f"程序执行过程中发生严重错误: {e}")
            logging.error(f"程序执行过程中发生严重错误: {e}")
        finally:
            if not self.args.keep_model_loaded:
                self.cleanup()
    
    def cleanup(self):
        """清理模型和释放内存"""
        try:
            if self.model is not None:
                del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"清理资源时发生异常: {e}")

def main():
    parser = argparse.ArgumentParser(description="Keye-VL媒体描述生成工具")
    
    # 必需参数
    parser.add_argument("source_path", help="输入媒体文件路径或包含媒体文件的文件夹路径")
    parser.add_argument("output_dir", help="输出目录路径")
    
    # 模型参数
    parser.add_argument("--model_path", default="Kwai-Keye/Keye-VL-1_5-8B",
                       help="Keye-VL模型路径")
    parser.add_argument("--use_flash_attention", action="store_true",
                       help="是否使用flash attention加速")
    
    # 处理参数
    parser.add_argument("--text", default="请描述这个内容",
                       help="描述提示文本，可添加/think或/no_think后缀指定模式")
    parser.add_argument("--max_duration", type=float, default=10.0,
                       help="最大处理视频时长(秒)，-1表示无限制")
    parser.add_argument("--fps", type=float, default=1.0,
                       help="视频采样帧率")
    parser.add_argument("--max_frames", type=int, default=16,
                       help="最大处理帧数")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度（0-1）")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                       help="最大新生成token数量")
    
    # 其他参数
    parser.add_argument("--keep_model_loaded", action="store_true",
                       help="处理完成后保持模型加载状态")
    
    args = parser.parse_args()
    
    # 创建处理器并处理媒体文件
    captioner = KeyeVL_Captioner(args)
    captioner.process_all_media()

if __name__ == "__main__":
    main()
