import argparse
from datetime import date
import time, subprocess
from ollama import Client

from FoodSAM_tools.db_operation import get_newest_health_data, get_user_nutrition_intake, login_and_get_id, insert_or_update_nutrition, get_recommand_intake, find_user_id
def open_ollama(email, retries=3, delay=5):
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    
    command = "ollama serve"
    attempt = 0
    process = None

    
    while attempt < retries:
        try:
            print("Starting ollama server...")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)  # 確保伺服器完全啟動

            print("Generating advice...")
            generate_advice(email)
            print("Advice generated successfully.")

            break  # 成功時退出重試循環
        except Exception as e:
            print(f"An error occurred: {e}")
            attempt += 1
            print(f"Retrying... ({attempt}/{retries})")
            time.sleep(delay)
        finally:
            if process:
                process.terminate()
                process.wait()

    if attempt == retries:
        print("Max retries reached. Failed to execute `open_ollama`.")

def get_nutrition_intake(email):
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    data = get_user_nutrition_intake(email, today_str)
    id = find_user_id(email)
    if data is None:
        data = (id[0], today_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    data = list(data)
    return data

def generate_advice(email):
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    health_data = get_newest_health_data(email)
    if health_data is None:
        health_data = ['Null', 4.5, 4.5, 120, 90, 100, 100]
    nutrition_data = get_nutrition_intake(email)
    recommand_intake = get_recommand_intake(email)
    
    ch_prompt = f"""
        你現在扮演一位專業營養師，我是一位腎臟病患者。請用正體中文回答，簡短分析就好，不用太長，三個句子以內就好(重要！)，請務必給我建議不要說無法給建議(重要！重要！)。
        我今天目前攝取的營養素累計：
              - 熱量 (calories): {nutrition_data[2]} kcal
              - 碳水化合物 (carbohydrate): {nutrition_data[3]} g
              - 蛋白質 (protein): {nutrition_data[4]} g
              - 脂肪 (fat): {nutrition_data[5]} g
              - 鉀 (K): {nutrition_data[6]} mg
              - 磷 (P): {nutrition_data[7]} mg
              - 鈉 (Na): {nutrition_data[12]} mg
              - 鈣 (Ca): {nutrition_data[13]} mg
              - 維生素A (Vitamin A): {nutrition_data[8]} µg
              - 維生素D (Vitamin D): {nutrition_data[9]} µg
              - 維生素E (Vitamin E): {nutrition_data[10]} mg
              - 維生素K (Vitamin K): {nutrition_data[11]} µg

              我的每日所需營養素建議：
              - 熱量 (calories): {recommand_intake[0]} kcal
              - 碳水化合物 (carbohydrate): {recommand_intake[1]} g
              - 蛋白質 (protein): {recommand_intake[2]} g
              - 脂肪 (fat): {recommand_intake[3]} g
              - 磷 (P): 700 mg
              - 鈉 (Na): 2400 mg
              - 鈣 (Ca): {recommand_intake[4]} mg
              - 維生素A (Vitamin A): 800 µg
              - 維生素D (Vitamin D): 15 µg
              - 維生素E (Vitamin E): 15 mg
              - 維生素K (Vitamin K): 100 µg

              請幫我設計我下一餐的健康餐盒菜色：
              菜單：
              - 主食：
              - 副食一：
              - 副食二：
              - 副食三：
    """
    client = Client(host='http://localhost:11434')
    response = client.chat(model='llama3.1:8b-instruct-fp16', messages=[       #current model list: llama2, llama3.1:70b, llama3.2, llama3.1:70b-instruct-q4_K_S, llama3.1:8b-instruct-fp16, llama3.2:3b-instruct-fp16
        {
            'role': 'user',
            'content': ch_prompt
        },
        ],
    )
    advice_save_path = f'/workspace/xien/FoodSAM/generation_text/{email}.txt'
    with open(advice_save_path, 'w') as text_file:
        text_file.write(response['message']['content'])

def main():
    parser = argparse.ArgumentParser(description="Run inference using FoodSAM")
    
    # 添加 --email 參數
    parser.add_argument("--email", type=str, required=True, help="User email for authentication or logging purposes")
    
    # 解析參數
    args = parser.parse_args()

    # 使用 args.email
    open_ollama(args.email)


if __name__ == "__main__":
    main()
