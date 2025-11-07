import cv2
import os
def images_to_mp4(image_folder, output_path, fps=10):
    """
    将图片序列转换为MP4视频

    Args:
        image_folder: 图片文件夹路径
        output_path: 输出视频路径
        fps: 帧率
    """
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # 按文件名排序

    if not images:
        print("没有找到PNG图片")
        return

    # 读取第一张图片获取尺寸
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入每一帧
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        print(f"已处理: {image}")

    video.release()
    print(f"视频已保存到: {output_path}")


# 在代码最后调用
if __name__ == "__main__":
    image_folder = r"C:\Users\Amine\Desktop\fuzzy_mpc\pic"
    output_path = r"C:\Users\Amine\Desktop\fuzzy_mpc\pic\simulation_video.mp4"
    images_to_mp4(image_folder, output_path, fps=12)