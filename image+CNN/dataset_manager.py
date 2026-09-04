"""
数据集管理模块
支持自动下载和加载 DIV2K、BSD68、SET12 等标准数据集
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm


class DatasetManager:
    """数据集下载和管理"""
    
    def __init__(self, data_dir: str = 'datasets'):
        """
        初始化数据集管理器
        
        Args:
            data_dir: 数据集保存目录
        """
        self.data_dir = Path(data_dir)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            if not self.data_dir.exists():
                raise

    def _is_writable(self, path: Path) -> bool:
        """检查路径是否可写"""
        try:
            return os.access(str(path), os.W_OK)
        except Exception:
            return False

    def _find_existing_div2k(self, subset: str, max_images: Optional[int]) -> List[str]:
        """
        尝试在本地已有目录中查找 DIV2K 图像
        支持以下结构：
        - data_dir 指向 .../DIV2K_train_HR 或 .../DIV2K_valid_HR
        - data_dir/DIV2K_train_HR
        - data_dir/DIV2K_valid_HR
        - data_dir/DIV2K/train_HR 或 data_dir/DIV2K/valid_HR
        - data_dir/train_HR 或 data_dir/valid_HR
        """
        candidates = []

        if self.data_dir.name.lower().endswith('_hr'):
            candidates.append(self.data_dir)

        candidates.extend([
            self.data_dir / f'DIV2K_{subset}_HR',
            self.data_dir / 'DIV2K' / f'{subset}_HR',
            self.data_dir / f'{subset}_HR',
        ])

        for subset_dir in candidates:
            if subset_dir.exists():
                images = sorted([str(p) for p in subset_dir.glob('*.png')])
                if images:
                    if max_images:
                        images = images[:max_images]
                    print(f"  使用本地已存在目录: {subset_dir}")
                    print(f"  ✓ 加载了 {len(images)} 张 DIV2K 图像")
                    return images

        return []
    
    def download_div2k(self, subset: str = 'train', max_images: Optional[int] = None) -> List[str]:
        """
        下载 DIV2K 数据集（HR图像）
        
        Args:
            subset: 'train' 或 'valid'
            max_images: 最多下载多少张图像，None表示全部
            
        Returns:
            图像路径列表
        """
        print(f"下载 DIV2K ({subset}) 数据集...")

        # 先尝试读取本地已存在数据集
        existing = self._find_existing_div2k(subset=subset, max_images=max_images)
        if existing:
            return existing

        # 如果数据目录不可写，则直接返回
        if not self._is_writable(self.data_dir):
            print(f"  ✗ 数据目录不可写: {self.data_dir}")
            print("  请将 data_dir 指向已下载的 DIV2K_train_HR 或 DIV2K_valid_HR 目录")
            return []
        
        subset_dir = self.data_dir / 'DIV2K' / f'{subset}_HR'
        try:
            subset_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            print(f"  ✗ 无法创建目录: {subset_dir}")
            return []
        
        # 检查是否已下载
        existing_images = list(subset_dir.glob('*.png'))
        if existing_images:
            print(f"  找到 {len(existing_images)} 张已下载的图像")
            return sorted([str(p) for p in existing_images[:max_images]])
        
        # DIV2K 下载链接（原始源 + 备用源）
        base_url = f"http://data.cv.snu.ac.kr:8008/DIV2K/"
        # 备用源：使用国内镜像或其他源
        backup_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
        
        if subset == 'train':
            filename = "DIV2K_train_HR.zip"
            url = base_url + filename
            backup_url = backup_url + filename
            num_images = 800
        else:
            filename = "DIV2K_valid_HR.zip"
            url = base_url + filename
            backup_url = backup_url + filename
            num_images = 100
        
        zip_path = self.data_dir / filename
        
        try:
            # 下载
            if not zip_path.exists():
                print(f"  下载 {filename} ({num_images} 张图像)...")
                try:
                    # 尝试原始源
                    self._download_file(url, str(zip_path))
                except Exception as e:
                    # 如果失败，尝试备用源
                    print(f"  原始源下载失败，尝试备用源...")
                    self._download_file(backup_url, str(zip_path))
            
            # 解压
            if not (subset_dir / '0001.png').exists():
                print(f"  解压到 {subset_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir / 'DIV2K')
                
                # 重新组织目录
                src_dir = self.data_dir / 'DIV2K' / f'DIV2K_{subset}_HR'
                if src_dir.exists():
                    for img in src_dir.glob('*.png'):
                        img.rename(subset_dir / img.name)
                    src_dir.rmdir()
            
            images = sorted([str(p) for p in subset_dir.glob('*.png')])
            if max_images:
                images = images[:max_images]
            
            print(f"  ✓ 加载了 {len(images)} 张 DIV2K 图像")
            return images
            
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print(f"  请手动下载: {url}")
            return []
    
    def download_bsd68(self) -> List[str]:
        """
        下载 BSD68 测试数据集
        
        Returns:
            图像路径列表
        """
        print("下载 BSD68 数据集...")
        
        subset_dir = self.data_dir / 'BSD68'
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已下载
        existing_images = list(subset_dir.glob('*.png'))
        if existing_images:
            print(f"  找到 {len(existing_images)} 张已下载的图像")
            return sorted([str(p) for p in existing_images])
        
        # BSD68 通常来自 BSD 数据集
        url = "https://www.cs.columbia.edu/DVMM/data/Berkeley_segmentation_dataset_500.tgz"
        
        try:
            tar_path = self.data_dir / 'bsd.tgz'
            
            if not tar_path.exists():
                print(f"  下载 BSD68 数据集...")
                self._download_file(url, str(tar_path))
            
            if not (subset_dir / '1.jpg').exists():
                print(f"  解压到 {subset_dir}...")
                with tarfile.open(tar_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(self.data_dir)
                
                # 复制图像
                src_dir = self.data_dir / 'BerkeleySegmentationDataset_500' / 'data' / 'images'
                if src_dir.exists():
                    for i, img in enumerate(sorted(src_dir.glob('*.jpg'))[:68], 1):
                        dst = subset_dir / f'{i:03d}.png'
                        if not dst.exists():
                            img_array = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
                            cv2.imwrite(str(dst), img_array)
            
            images = sorted([str(p) for p in subset_dir.glob('*.png')])
            print(f"  ✓ 加载了 {len(images)} 张 BSD68 图像")
            return images
            
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            return []
    
    def download_set12(self) -> List[str]:
        """
        下载 SET12 标准测试集
        
        Returns:
            图像路径列表
        """
        print("下载 SET12 数据集...")
        
        subset_dir = self.data_dir / 'SET12'
        subset_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已下载
        existing_images = list(subset_dir.glob('*.png'))
        if existing_images:
            print(f"  找到 {len(existing_images)} 张已下载的图像")
            return sorted([str(p) for p in existing_images])
        
        # SET12 标准图像列表（来自公开资源）
        set12_images = [
            ('barbara', 'http://www.cs.tut.fi/~foi/GCF-BM3D/barbara.png'),
            ('boats', 'http://www.cs.tut.fi/~foi/GCF-BM3D/boats.png'),
            ('cameraman', 'http://www.cs.tut.fi/~foi/GCF-BM3D/cameraman.png'),
            ('couple', 'http://www.cs.tut.fi/~foi/GCF-BM3D/couple.png'),
            ('fingerprint', 'http://www.cs.tut.fi/~foi/GCF-BM3D/fingerprint.png'),
            ('foreman', 'http://www.cs.tut.fi/~foi/GCF-BM3D/foreman.png'),
            ('hill', 'http://www.cs.tut.fi/~foi/GCF-BM3D/hill.png'),
            ('lena', 'http://www.cs.tut.fi/~foi/GCF-BM3D/lena.png'),
            ('man', 'http://www.cs.tut.fi/~foi/GCF-BM3D/man.png'),
            ('montage', 'http://www.cs.tut.fi/~foi/GCF-BM3D/montage.png'),
            ('peppers', 'http://www.cs.tut.fi/~foi/GCF-BM3D/peppers.png'),
            ('starfish', 'http://www.cs.tut.fi/~foi/GCF-BM3D/starfish.png'),
        ]
        
        try:
            count = 0
            for name, url in set12_images:
                img_path = subset_dir / f'{name}.png'
                
                if not img_path.exists():
                    try:
                        self._download_file(url, str(img_path))
                        count += 1
                    except Exception as e:
                        print(f"    ✗ 下载 {name} 失败: {e}")
                else:
                    count += 1
            
            images = sorted([str(p) for p in subset_dir.glob('*.png')])
            print(f"  ✓ 加载了 {len(images)} 张 SET12 图像 (成功: {count}/12)")
            return images
            
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            return []
    
    def _download_file(self, url: str, save_path: str, chunk_size: int = 8192):
        """
        下载文件，带进度条
        
        Args:
            url: 下载链接
            save_path: 保存路径
            chunk_size: 块大小
        """
        try:
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('Content-Length', 0))
            
            downloaded = 0
            with open(save_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = 100 * downloaded / total_size
                        print(f"    下载进度: {percent:.1f}%", end='\r')
            
            print()  # 新行
        except Exception as e:
            print(f"    下载错误: {e}")
            raise
    
    def load_images(self, image_paths: List[str], max_size: int = 512,
                   grayscale: bool = True) -> List[np.ndarray]:
        """
        加载图像列表
        
        Args:
            image_paths: 图像路径列表
            max_size: 最大尺寸（保持长宽比）
            grayscale: 是否转换为灰度图
            
        Returns:
            图像 numpy 数组列表
        """
        images = []
        
        for path in image_paths:
            try:
                if grayscale:
                    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # 缩放到最大尺寸
                h, w = img.shape[:2]
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # 转换为 float32，范围 [0, 1]
                img = img.astype(np.float32) / 255.0
                images.append(img)
                
            except Exception as e:
                print(f"  加载失败 {path}: {e}")
                continue
        
        return images


def prepare_real_dataset(dataset_name: str = 'div2k_train', max_images: Optional[int] = None,
                        data_dir: str = 'datasets') -> Tuple[List[str], List[np.ndarray]]:
    """
    准备真实数据集
    
    Args:
        dataset_name: 数据集名称
            - 'div2k_train': DIV2K 训练集 (800 张)
            - 'div2k_valid': DIV2K 验证集 (100 张)
            - 'bsd68': BSD68 测试集 (68 张)
            - 'set12': SET12 标准集 (12 张)
        max_images: 最多加载多少张图像
        data_dir: 数据集保存目录
        
    Returns:
        (图像路径列表, 加载的图像数组列表)
    """
    manager = DatasetManager(data_dir)
    
    if dataset_name.startswith('div2k'):
        subset = 'valid' if 'valid' in dataset_name else 'train'
        image_paths = manager.download_div2k(subset=subset, max_images=max_images)
    elif dataset_name == 'bsd68':
        image_paths = manager.download_bsd68()
        if max_images:
            image_paths = image_paths[:max_images]
    elif dataset_name == 'set12':
        image_paths = manager.download_set12()
        if max_images:
            image_paths = image_paths[:max_images]
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    if not image_paths:
        print(f"警告: {dataset_name} 无法获取图像")
        return [], []
    
    print(f"加载图像...")
    images = manager.load_images(image_paths)
    
    print(f"✓ 成功加载 {len(images)} 张图像")
    return image_paths, images
