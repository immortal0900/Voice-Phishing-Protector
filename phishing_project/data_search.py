import os
import shutil

source_folder = r"C:\normal_sound"
destination_folder = r"C:\phishing_project\data\normal"

os.makedirs(destination_folder, exist_ok=True)
#exist_ok=True 폴더가 이미 있어도 에러가 발생 안함


for root, dirs, files in os.walk(source_folder):
# os.walk를 사용하여 source_folder와 그 안의 하위 폴더를 탐색함    
    for file in files:
        if file.endswith(".json"):
            source_path = os.path.join(root, file)
            # 원본 파일의 전체 경로를 만든다.
            # 예: C:\normal_sound\S000001\sample.json
            shutil.copy(source_path, destination_folder)
            # 원본 파일을 목적지 폴더에 복사함

            print(f" 복사완료 : {source_path}")
print("모든 .json파일 복사완료")









