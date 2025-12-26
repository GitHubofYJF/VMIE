import numpy as np
import torch
import torch.nn as nn
from model import Glow # model.py文件
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
from PIL import Image 
from torchvision import transforms
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
import os
import vector_chao_map # vector_chao_map.py文件
import time
import random
import DC_AE     # DC_AE.py文件
import test
import attack # attack.py文件
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def calc_z_shapes(n_channel, input_size, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes

def split_z(z_seq, n_channel, input_size, n_block): 

    z_sample = []

    if len(z_seq.shape) == 1:
        z_seq = z_seq.reshape(((1,) + z_seq.shape))
    n_sample = len(z_seq)
    z_seq = z_seq

    z_shapes = calc_z_shapes(n_channel, input_size, n_block)
    start = 0
    for single_shape in z_shapes:
        length = 1
        for _ in single_shape:
            length *= _
        temp = z_seq[:, start:start + length]
        temp = temp.reshape((n_sample,) + single_shape)
        z_new = torch.from_numpy(temp)
        z_sample.append(z_new.to(device))
        start += length
    return z_sample

def concat_z(z_list):   # 将多个z张量串联成一个。
    z_seq = None
    for i, z in enumerate(z_list):
        temp = z.cpu().data.numpy()
        temp = temp.reshape(temp.shape[0], -1)
        if i == 0:
            z_seq = temp
        else:
            z_seq = np.concatenate((z_seq, temp), axis=1)
    return z_seq

def x2img(x, bit=8):    # 根据指定的位深度将浮点数数组转换为图像
    if bit == 16:
        img = (np.clip(x + 0.5, 0, 1) * 65535).astype(np.uint16)
        return img
    elif bit == 8:
        img = (np.clip(x + 0.5, 0, 1) * 255).astype(np.uint8)
        return img
    else:
        print("Undefine bit")
        return x

def generateimage():
    # 将原始图像数组添加到一个批处理维度
    image_size=128
    totalnum = image_size * image_size * image_channel#计算图像的像素总数。
    outdir="generate_img"
    os.makedirs(outdir, exist_ok=True)
    resnum =totalnum
    for i in range(1,1201):
        seed_np=i
        print(seed_np)
        z2 = np.random.RandomState(seed_np).randn(1, resnum)#随机生成向量z2
        z2=z2.astype(np.float32)*0.7
        z=[]
        z.append(z2)
        z_input = split_z(z2, image_channel, image_size, n_block)#分块

        #glow模型生成图片
        float_img = glow.reverse(z_input).cpu().data#编码的潜在向量通过glow.reverse()函数，生成浮点图像
        #数据预处理，显示图像
        plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0))#数据转换为numpy数组，.squeeze(0)表示移除大小为1的维度，.transpose((1, 2, 0))表示进行转置。
        stego_image = x2img(plot_img, 8)#行代码调用一个名为x2img的函数，将数据转换为图像
        stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)#使用OpenCV库将图像从RGB颜色空间转换为BGR颜色空间。        
        save_path = f'{outdir}/{i}.png'#设定保存路径
        cv2.imwrite(save_path, stego_image) 

def get_image_files(folder_path):
    """获取文件夹中所有常见格式的图片文件"""
    # 常见图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    
    image_files = []
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            # 确保是文件而不是文件夹
            if os.path.isfile(file_path):
                image_files.append(file_path)
    
    return image_files

def process_image(image_path):
    """处理单张图片的函数，可以根据需要修改"""
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 示例处理1: 获取图片信息
            width, height = img.size
            format = img.format
            mode = img.mode
            
            # 示例处理2: 转换为灰度图
            # gray_img = img.convert('L')
            
            # 示例处理3: 缩小图片
            # resized_img = img.resize((width//2, height//2))
            
            print(f"处理图片: {os.path.basename(image_path)}")
            print(f"  格式: {format}, 尺寸: {width}x{height}, 模式: {mode}")
            
            # 如果需要保存处理后的图片
            # output_path = os.path.splitext(image_path)[0] + "_processed." + format.lower()
            # gray_img.save(output_path)
            # print(f"  已保存处理后的图片到: {output_path}")
            
            return True
    
    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {str(e)}")
        return False

def process_all_images_in_folder(folder_path):
    """处理文件夹中的所有图片"""
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return
    
    # 获取所有图片文件
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"在文件夹 '{folder_path}' 中未找到任何图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...\n")
    
    # 在for循环中依次处理每个图片
    for image_path in image_files:
        process_image(image_path)
        # 可以在这里添加每个图片处理后的其他操作
    
    print(f"\n处理完成，共处理了 {len(image_files)} 个图片文件")
def L_compare(L):
    for i in range(1,4):
        for j in range(1,3):
            img_path=f"L{L}_im{i}_{j}.png"
            if j==1:
                seed=None
            else:
                seed=None
            image_path = f"im_{i}.jpg"
            image = Image.open(image_path).convert('RGB').resize((input_size, input_size))
            image_np = np.array(image)  # 转换为 NumPy 数组 (H, W, C)
            
            # -------------------- 编码阶段 --------------------
            image_tensor = torch.from_numpy(image_np).permute(2 ,0, 1).float() / 255.0  # [C, H, W]
            image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, 64, 64]
            # 编码器输出
            with torch.no_grad():
                encoded = DCAE.encoder(image_batch)  # 输出 [1, 40, 16, 16]
            encoded_np = encoded.squeeze(0).cpu().numpy().astype(np.float32)
            num = encoded_np.size   # 获取处理后的图像 z1 的像素数量。
            z1= encoded_np*0.7
            z1.resize(1,num)
            totalnum = image_size  * image_size  * image_channel    # 计算图像的像素总数。
            resnum =totalnum - num

            z2 = np.random.RandomState(seed).randn(1, resnum)   # 随机生成向量z2
            # print(z2)
            z2=z2.astype(np.float32)
            z2=z2*0.7
            
            tz = np.concatenate((z1, z2), axis=1)    # z1,z2列拼接
            tempz =vector_chao_map.encryted_128(tz,3.9999,0.666,5000,5000) 

            z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
            float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像

            plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
            # plot_img = float_img.numpy()
            stego_image = x2img(plot_img, 8)    # 数据转换为图像
            stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
            # plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint8))
            # plt.show()    

            save_dir="L"
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, img_path)
            cv2.imwrite(filename, stego_image)
   
            _, _, ext_z_input = glow1.forward(float_img.to(device)) # glow模型逆处理图片得到输入向量

            ext_z = concat_z(ext_z_input)
            ext_z = ext_z[0]

            re_partz = ext_z[0:totalnum]

            re_partz =vector_chao_map.decryted_128(re_partz ,3.9999,0.666,5000,5000)
            re_partz = re_partz[0:num]
            re_partz=re_partz
            re = re_partz.reshape((1, L, 16, 16))#调整张量
            # re = re_partz.reshape((1, 160, 16, 16))#调整张量
            # re = re_partz.reshape((128,128, 3))
            
            re=torch.from_numpy(re).float().to(device)
            with torch.no_grad():
                recon_img = DCAE.decoder(re)
            
            transform = transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor()
            ])
            
            # 加载测试图像
            img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
            # img = image_np.to(device)  # 确保 img 是张量且已加载到设备
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_np = recon_img  .squeeze(0).permute(1, 2, 0).cpu().numpy()


            img_uint8 = (img_np * 255).astype(np.uint8)
            recon_uint8 = (recon_np * 255).astype(np.uint8)

            # 转换颜色通道顺序：RGB → BGR（如果模型输出是RGB）
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            recon_bgr = cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2BGR)
            # save_dir="attcak_img"
            # os.makedirs(save_dir, exist_ok=True)
            # cv2.imwrite('attcak_img/origin_img.png', img_bgr )
            # filename = os.path.join(save_dir, img_path)
            # cv2.imwrite(filename , recon_bgr)
            # cv2.imwrite('decrypted.png', recon_bgr)
            # 计算指标
            psnr = peak_signal_noise_ratio(img_np, recon_np, data_range=1.0)
            ssim = structural_similarity(img_np, recon_np, data_range=1.0, channel_axis=2)
            print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
          
if __name__ == '__main__':
    #参数设置
    image_size = 128 # 生成图像尺寸
    input_size= 128 # 自编码器的输入尺寸，影响重构质量
    image_channel=3
    n_flow = 32
    n_block = 4


    glow = Glow(in_channel=image_channel, n_flow=n_flow, n_block=n_block, affine=False, conv_lu=True)
    glow1 = glow.to(device)
    glow1.load_state_dict(torch.load(f"model_0400001_single.pt",map_location=torch.device(device)))

    
    # DCAE = DC_AE.DCAE_128to128().to(device)
    # DCAE.eval()
    # DCAE.load_state_dict(torch.load("dcae_128.pth",map_location=torch.device(device)))

    # DCAE = DC_AE.DCAE_256to128_Middle().to(device)
    # DCAE.eval()
    # DCAE.load_state_dict(torch.load("dcae_256.pth",map_location=torch.device(device)))

    # DCAE = DC_AE.DCAE_256to128().to(device)
    # DCAE.eval()
    # DCAE.load_state_dict(torch.load("dcae_256to128.pth",map_location=torch.device(device)))


    L=180
    DCAE = DC_AE.L180().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load(f"L{L}.pth",map_location=torch.device(device)))
    #  L_compare(L)

    L=188
    DCAE = DC_AE.L188().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load(f"L{L}.pth",map_location=torch.device(device)))
    # L_compare(L)
    L=128
    DCAE = DC_AE.L128_2().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load(f"L/L{L}.pth",map_location=torch.device(device)))
    '''
    image_size=128
    totalnum = image_size * image_size * image_channel#计算图像的像素总数。
    outdir="generate_img"
    os.makedirs(outdir, exist_ok=True)
    resnum =totalnum
    for i in range(1,1201):
        seed_np=i
        print(seed_np)
        z2 = np.random.RandomState(seed_np).randn(1, resnum)#随机生成向量z2
        z2=z2.astype(np.float32)*0.7
        z=[]
        z.append(z2)
        z_input = split_z(z2, image_channel, image_size, n_block)#分块

        #glow模型生成图片
        float_img = glow.reverse(z_input).cpu().data#编码的潜在向量通过glow.reverse()函数，生成浮点图像
        #数据预处理，显示图像
        plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0))#数据转换为numpy数组，.squeeze(0)表示移除大小为1的维度，.transpose((1, 2, 0))表示进行转置。
        stego_image = x2img(plot_img, 8)#行代码调用一个名为x2img的函数，将数据转换为图像
        stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)#使用OpenCV库将图像从RGB颜色空间转换为BGR颜色空间。        
        save_path = f'{outdir}/{i}.png'#设定保存路径
        cv2.imwrite(save_path, stego_image) 
    '''
    '''
    DCAE = DC_AE.DCAE_128to128_new().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load("dcae_128_1.pth",map_location=torch.device(device)))
    '''
    '''
    DCAE = DC_AE.DCAE_512to128().to(device)
    DCAE.eval()
    DCAE.load_state_dict(torch.load("dcae_512.pth",map_location=torch.device(device)))
    '''
    '''
    folder_path='D:\\workspace\\val2017_cover'

    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
    
    
    # 获取所有图片文件
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print(f"在文件夹 '{folder_path}' 中未找到任何图片文件")
   
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...\n")
    outdir="glow_32"
    os.makedirs(outdir, exist_ok=True)
    i=0
    for image_path in image_files:
        i=i+1
        print(i)
        if i>1100:
            break
        image = Image.open(image_path).convert('RGB').resize((input_size, input_size))
        image_np = np.array(image)  # 转换为 NumPy 数组 (H, W, C)
        
        image_tensor = torch.from_numpy(image_np).permute(2 ,0, 1).float() / 255.0  # [C, H, W]
        image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, 64, 64]
        # 编码器输出
        with torch.no_grad():
            encoded = DCAE.encoder(image_batch)  # 输出 [1, 40, 16, 16]
        encoded_np = encoded.squeeze(0).cpu().numpy().astype(np.float32)
        num = encoded_np.size   
        z1= encoded_np*0.7
        z1.resize(1,num)
        totalnum = image_size  * image_size  * image_channel 
        resnum =totalnum - num
        seed=random.seed(time.time())

        # seed=42
        z2 = np.random.RandomState(seed).randn(1, resnum)   

        z2=z2.astype(np.float32)
        z2=z2*0.7
        
        tempz = np.concatenate((z1, z2), axis=1)   

        z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
        float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像

        #数据预处理，显示图像
        plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0))#数据转换为numpy数组，.squeeze(0)表示移除大小为1的维度，.transpose((1, 2, 0))表示进行转置。
        stego_image = x2img(plot_img, 8)#行代码调用一个名为x2img的函数，将数据转换为图像
        stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)#使用OpenCV库将图像从RGB颜色空间转换为BGR颜色空间。   
        save_path = f'{outdir}/{i}.png'#设定保存路径
        cv2.imwrite(save_path, stego_image) 

    print(f"\n处理完成，共处理了 {len(image_files)} 个图片文件")
    '''
    ###----------------------------------------------------------------------------------------------

    image_path = "cat.png"
    image = Image.open(image_path).convert('RGB').resize((input_size, input_size))
    image_np = np.array(image)  # 转换为 NumPy 数组 (H, W, C)
 
    # -------------------- 编码阶段 --------------------
    image_tensor = torch.from_numpy(image_np).permute(2 ,0, 1).float() / 255.0  # [C, H, W]
    image_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, 64, 64]
    # 编码器输出
    with torch.no_grad():
        encoded = DCAE.encoder(image_batch)  # 输出 [1, 40, 16, 16]
    encoded_np = encoded.squeeze(0).cpu().numpy().astype(np.float32)
    num = encoded_np.size   # 获取处理后的图像 z1 的像素数量。
    z1= encoded_np
    z1.resize(1,num)
    totalnum = image_size  * image_size  * image_channel    # 计算图像的像素总数。
    resnum =totalnum - num
    seed=random.seed(time.time())
    # print(z1)
    
    z2 = np.random.RandomState(seed).randn(1, resnum)   # 随机生成向量z2
    # print(z2)
    z2=z2.astype(np.float32)
    z2=z2*0.7
    '''
    tz = np.concatenate((z1, z2), axis=1)    # z1,z2列拼接
    tempz =vector_chao_map.encryted_128(tz,3.9999,0.666,5000,5000) # 图像编码数据加密 3.999,0.667,5000,5000

    """
    KEY =3.999,0.666,5000,5000
    KEY1=4,0.666,5000,5000
    KEY2=3.999,0.667,5000,5000
    KEY3=3.999,0.666,5001,5000
    KEY4=3.999,0.666,5000,4999
    """

    z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
    float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像
    # float_img=attack.crop_and_pad_attack(float_img.squeeze(0), crop_ratio=0.5, crop_mode='random', pad_mode='zeros')   # 'center' 裁剪
    # float_img= float_img.unsqueeze(0) # 0.1 0.3 0.5
    # 显示图像
    # float_img =attack.add_salt_pepper_noise_tensor(float_img , prob=0.0001)# 0.001 0.0005 0.0001
    # float_img = attack.add_gaussian_noise(float_img, mean=0, std=10/255.0, inplace=False)
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
    # plot_img = float_img.numpy()

    stego_image = x2img(plot_img, 16)    # 数据转换为图像
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint16))
    plt.show()   
    print(float_img.shape)
    print(plot_img.max())
    print(plot_img.min())
    scale = 10**3  # 1000000
    plot_img = (plot_img * scale).astype(np.int64) / scale
    print(plot_img.max())
    print(plot_img.min())
        # NumPy → Tensor，形状 [H, W, C]
    float_img = torch.from_numpy(plot_img)
    float_img = float_img.permute(2, 0, 1)  # 转置维度：[H,W,C] → [C,H,W]（Tensor 用 permute，NumPy 用 transpose）
    float_img = float_img.unsqueeze(0)      # 添加 batch 维度：[C,H,W] → [1,C,H,W]
    float_img = float_img.to(dtype=torch.float32)  # 确保数据类型是 float32
    save_dir="L"
    print(float_img.shape)
    img_path="RGB16.png"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, img_path)
    print(stego_image.dtype)
    print(stego_image.shape)
    print(stego_image.ndim)
    # output_img = Image.fromarray(stego_image, mode='RGB;16')
    # output_img.save(img_path, 'PNG', bitdepth=16)
    cv2.imwrite(img_path, stego_image)
    # stego_image.save('RGB16.png', 'PNG', bitdepth=16)
    '''
    '''
    # -------------------- 秘钥一 --------------------

    tempz =vector_chao_map.encryted_128(tz,4,0.666,5000,5000) # 图像编码数据加密 3.999,0.667,5000,5000
    save_dir="save_img"
    # img_path="attcak_3_gaussian_10.png"
    img_path="key21.png"

    z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
    float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像
    # float_img=attack.crop_and_pad_attack(float_img.squeeze(0), crop_ratio=0.5, crop_mode='random', pad_mode='zeros')   # 'center' 裁剪
    # float_img= float_img.unsqueeze(0) # 0.1 0.3 0.5
    # 显示图像
    # float_img =attack.add_salt_pepper_noise_tensor(float_img , prob=0.0001)# 0.001 0.0005 0.0001
    # float_img = attack.add_gaussian_noise(float_img, mean=0, std=10/255.0, inplace=False)
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
    # plot_img = float_img.numpy()
    stego_image = x2img(plot_img, 8)    # 数据转换为图像
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint8))
    plt.show()    


    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, img_path)
    cv2.imwrite(filename, stego_image)

    # -------------------- 秘钥二 --------------------

    tempz =vector_chao_map.encryted_128(tz,3.999,0.667,5000,5000) # 图像编码数据加密 3.999,0.667,5000,5000
    save_dir="save_img"
    # img_path="attcak_3_gaussian_10.png"
    img_path="key22.png"

    z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
    float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像
    # float_img=attack.crop_and_pad_attack(float_img.squeeze(0), crop_ratio=0.5, crop_mode='random', pad_mode='zeros')   # 'center' 裁剪
    # float_img= float_img.unsqueeze(0) # 0.1 0.3 0.5
    # 显示图像
    # float_img =attack.add_salt_pepper_noise_tensor(float_img , prob=0.0001)# 0.001 0.0005 0.0001
    # float_img = attack.add_gaussian_noise(float_img, mean=0, std=10/255.0, inplace=False)
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
    # plot_img = float_img.numpy()
    stego_image = x2img(plot_img, 8)    # 数据转换为图像
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint8))
    plt.show()    


    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, img_path)
    cv2.imwrite(filename, stego_image)

# -------------------- 秘钥三 --------------------



    tempz =vector_chao_map.encryted_128(tz,3.999,0.666,5001,5000) # 图像编码数据加密 3.999,0.667,5000,5000
    save_dir="save_img"
    # img_path="attcak_3_gaussian_10.png"
    img_path="key23.png"

    z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
    float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像
    # float_img=attack.crop_and_pad_attack(float_img.squeeze(0), crop_ratio=0.5, crop_mode='random', pad_mode='zeros')   # 'center' 裁剪
    # float_img= float_img.unsqueeze(0) # 0.1 0.3 0.5
    # 显示图像
    # float_img =attack.add_salt_pepper_noise_tensor(float_img , prob=0.0001)# 0.001 0.0005 0.0001
    # float_img = attack.add_gaussian_noise(float_img, mean=0, std=10/255.0, inplace=False)
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
    # plot_img = float_img.numpy()
    stego_image = x2img(plot_img, 8)    # 数据转换为图像
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint8))
    plt.show()    


    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, img_path)
    cv2.imwrite(filename, stego_image)



# -------------------- 秘钥四 --------------------

    tempz =vector_chao_map.encryted_128(tz,3.999,0.666,5000,4999) # 图像编码数据加密 3.999,0.667,5000,5000
    save_dir="save_img"
    # img_path="attcak_3_gaussian_10.png"
    img_path="key24.png"

    z_input = split_z(tempz, image_channel, image_size, n_block)    # 分块
    float_img = glow1.reverse(z_input).cpu().data   # 生成浮点图像
    # float_img=attack.crop_and_pad_attack(float_img.squeeze(0), crop_ratio=0.5, crop_mode='random', pad_mode='zeros')   # 'center' 裁剪
    # float_img= float_img.unsqueeze(0) # 0.1 0.3 0.5
    # 显示图像
    # float_img =attack.add_salt_pepper_noise_tensor(float_img , prob=0.0001)# 0.001 0.0005 0.0001
    # float_img = attack.add_gaussian_noise(float_img, mean=0, std=10/255.0, inplace=False)
    plot_img = float_img.numpy().squeeze(0).transpose((1, 2, 0)) # 转换为numpy数组
    # plot_img = float_img.numpy()
    stego_image = x2img(plot_img, 8)    # 数据转换为图像
    stego_image = cv2.cvtColor(stego_image, cv2.COLOR_RGB2BGR)
    plt.imshow((np.clip(plot_img + 0.5, 0, 1) * 255).astype(np.uint8))
    plt.show()    


    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, img_path)
    cv2.imwrite(filename, stego_image)
    '''
    img_path="RGB16.png"
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    img= img.astype(np.float32) / 65535.0 -0.5
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    float_img = torch.from_numpy(img)
    float_img = float_img.permute(2, 0, 1)  # 转置维度：[H,W,C] → [C,H,W]（Tensor 用 permute，NumPy 用 transpose）
    float_img = float_img.unsqueeze(0)      # 添加 batch 维度：[C,H,W] → [1,C,H,W]
    float_img = float_img.to(dtype=torch.float32)

    _, _, ext_z_input = glow1.forward(float_img.to(device)) # glow模型逆处理图片得到输入向量
    ext_z = concat_z(ext_z_input)
    ext_z = ext_z[0]

    re_partz = ext_z[0:totalnum]

    re_partz =vector_chao_map.decryted_128(re_partz ,3.9999,0.666,5000,5000)
    re_partz = re_partz[0:num]
    re_partz=re_partz
    re = re_partz.reshape((1, L, 16, 16))#调整张量
    # re = re_partz.reshape((1, 160, 16, 16))#调整张量
    # re = re_partz.reshape((128,128, 3))
    
    re=torch.from_numpy(re).float().to(device)
    with torch.no_grad():
        recon_img = DCAE.decoder(re)
    
    transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop((input_size,input_size)),
        transforms.ToTensor()
    ])
    
    # 加载测试图像
    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
    # img = image_np.to(device)  # 确保 img 是张量且已加载到设备
    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = recon_img  .squeeze(0).permute(1, 2, 0).cpu().numpy()


    img_uint8 = (img_np * 255).astype(np.uint8)
    recon_uint8 = (recon_np * 255).astype(np.uint8)

    # 转换颜色通道顺序：RGB → BGR（如果模型输出是RGB）
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    recon_bgr = cv2.cvtColor(recon_uint8, cv2.COLOR_RGB2BGR)
    # save_dir="attcak_img"
    # os.makedirs(save_dir, exist_ok=True)
    # cv2.imwrite('attcak_img/origin_img.png', img_bgr )
    # filename = os.path.join(save_dir, img_path)
    # cv2.imwrite(filename , recon_bgr)
    # cv2.imwrite('decrypted.png', recon_bgr)
    # 计算指标
    psnr = peak_signal_noise_ratio(img_np, recon_np, data_range=1.0)
    ssim = structural_similarity(img_np, recon_np, data_range=1.0, channel_axis=2)
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    
    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Origin\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstruction")
    plt.imshow(recon_np)
    plt.axis('off')
    plt.show()

