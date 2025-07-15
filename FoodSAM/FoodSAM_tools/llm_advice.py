from ollama import Client
import os
import subprocess
import pandas as pd
from datetime import date
import time
import json
import torch
from FoodSAM_tools.db_operation import get_newest_health_data, get_user_nutrition_intake, login_and_get_id, insert_or_update_nutrition, get_recommand_intake, find_user_id
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
df_path = 'FoodSAM/FoodSAM_tools/nutrition_v3.csv'

def open_ollama(food_label, save_dir, email, label_pixel_count, retries=3, delay=5):
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    nutrition_intake(food_label, email, label_pixel_count, date=today_str)
    
    command = "ollama serve"
    attempt = 0
    process = None

    
    while attempt < retries:
        try:
            print("Starting ollama server...")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)  # 確保伺服器完全啟動

            print("Generating advice...")
            generate_advice(food_label, save_dir, email)
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
    

def generate_advice(food_label, save_dir, email):
    
    #names = ["background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom", "salad", "other ingredients"]
    names = ['背景', '糖果', '蛋撻', '薯條', '巧克力', '餅乾', '爆米花', '布丁', '冰淇淋', '奶酪黃油', '蛋糕', '葡萄酒', '奶昔', '咖啡', '果汁', '牛奶', '茶', '杏仁', '紅豆', '腰果', '蔓越莓乾', '黃豆', '核桃', '花生', '雞蛋', '蘋果', '棗子', '杏子', '酪梨', '香蕉', '草莓', '櫻桃', '藍莓', '覆盆子', '芒果', '橄欖', '桃子', '檸檬', '梨子', '無花果', '鳳梨', '葡萄', '奇異果', '哈密瓜', '橙子', '西瓜', '牛排', '豬肉', '雞鴨', '香腸', '炸肉', '羊肉', '醬料', '螃蟹', '魚', '貝類', '蝦', '湯', '麵包', '玉米', '漢堡', '披薩', '花捲包子', '餛飩水餃', '意大利麵', '麵條', '米飯', '派', '豆腐', '茄子', '馬鈴薯', '大蒜', '花椰菜', '番茄', '海帶', '海藻', '蔥', '油菜', '薑', '秋葵', '生菜', '南瓜', '黃瓜', '白蘿蔔', '胡蘿蔔', '蘆筍', '竹筍', '西蘭花', '芹菜', '香菜薄荷', '荷蘭豆', '捲心菜', '豆芽', '洋蔥', '胡椒', '四季豆', '法國豆', '杏鮑菇', '香菇', '金針菇', '牡蠣菇', '白蘑菇', '沙拉', '其他食材']

    food_prompt = ''
    for label in food_label:
      if label != 0:
        food_prompt += names[label]
        food_prompt += ', '

    
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    health_data = get_newest_health_data(email)
    if health_data is None:
        health_data = ['Null', 4.5, 4.5, 120, 90, 100, 100]
    out_of_range_count = 0
    if(health_data[1] > 4.7):
        out_of_range_count = out_of_range_count + 1
    if(health_data[2] > 5.5):
        out_of_range_count = out_of_range_count + 1
    if(health_data[3] > 125):
        out_of_range_count = out_of_range_count + 1
    if(health_data[4] > 130):
        out_of_range_count = out_of_range_count + 1
    if(health_data[5] > 160):
        out_of_range_count = out_of_range_count + 1
    
    if(out_of_range_count == 0):
        healthy = True
    else:
        healthy = False
    nutrition_data = get_nutrition_intake(email)
    recommand_intake = get_recommand_intake(email)
    food_prompt = food_prompt[:-2]    #remove last ", " 
    healthy_prompt = f"""
              你現在扮演一位專業營養師，我是一位健康使用者。請用正體中文回答，內容不超過150字元。 重要：不要太多字！不要太多字！不要太多字！

              我今天的飲食內容：{food_prompt}。
              
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

              最後，請根據我今天的飲食內容，提供幾點條列式的飲食建議。
              """
    ch_prompt_v2 = f"""
              你現在扮演一位專業營養師，我是一位腎臟病患者。請用正體中文回答，內容不超過150字元。 重要：不要太多字！不要太多字！不要太多字！

              我今天的抽血結果為：
              - 鉀 (K): {health_data[1]} mmol/L
              - 磷 (P): {health_data[2]} mg/dL
              - 隨機血糖 (Glucose random): {health_data[3]} mg/dL
              - 空腹血糖 (Glucose AC): {health_data[4]} mg/dL
              - 低密度脂蛋白膽固醇 (LDL-C): {health_data[5]} mg/dL
              - 腎絲球過濾率 (eGPR): {health_data[6]} ml/min/1.73m^2

              抽血結果建議正常值為：
              - 鉀 (K): 3.4-4.7 mmol/L
              - 磷 (P): 3.5-5.5 mg/dL
              - 隨機血糖 (Glucose random): 小於 125 mg/dL
              - 空腹血糖 (Glucose AC): 70-140 mg/dL
              - 低密度脂蛋白膽固醇 (LDL-C): 小於 130 mg/dL
              - 腎絲球過濾率 (eGPR): 大於 90 ml/min/1.73m^2

              我今天的飲食內容：{food_prompt}。
              
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

              - 鉀 (K): 請閱讀下方文章，用你學習到的知識以及我的腎絲球過濾率(eGPR)，來判斷我每日所需鉀重量為多少：

                CKD Stage 1-2 (eGPR >= 60 ml/min/1.73m^2)
                建議攝取量: 不限制鉀的攝取，除非血清鉀濃度升高。
                建議: KDOQI 2004建議每日至少4000毫克。患者若無高血鉀風險，應維持正常鉀攝取量。

                CKD Stage 3-4 (eGPR 30-59 ml/min/1.73m^2)
                建議攝取量: 每日鉀攝取量2000-4000毫克。
                建議: 對於CKD患者增加水果和蔬菜的攝取以減少體重、血壓和淨酸生成（KDOQI 2020），有助於延緩CKD進程。

                CKD Stage 3b-5 (非透析病患，eGPR < 45 ml/min/1.73m^2)
                建議攝取量: 每日鉀攝取量2000-3000毫克。
                建議: 若血清鉀水平超過5 mmol/L，則建議控制鉀含量高的食物，並建議採用浸泡或煮沸等食材處理方式以降低鉀含量。

                CKD Stage 5 (透析患者，eGPR < 15 ml/min/1.73m^2)
                透析患者建議攝取量:
                血液透析（HD）: 每日鉀攝取量2000-3000毫克。
                腹膜透析（PD）: 每日鉀攝取量2000-4000毫克。
                建議: 對透析患者使用低鉀透析液、延長透析時間、增加透析頻率等方式來改善鉀的去除效果。

                特別說明
                無鉀攝取指導共識: 目前對於CKD患者低鉀飲食的建議缺乏一致的指導。不同組織的建議有所不同，且缺乏強而有力的隨機對照試驗證據來支持嚴格的鉀攝取限制。
                漸進、個別化調整: 推薦進行個別化、漸進的鉀攝取減少，以確保CKD患者在避免高血鉀風險的同時，仍能攝取足量的新鮮水果與蔬菜。

              最後，請根據我的抽血結果和今天的飲食內容，提供幾點條列式的飲食建議。此外，根據每日所需營養素以及你所學的菜單知識為我設計下一餐（每個類別只選擇一道菜，不需解釋）：

              菜單：
              - 主食：
              - 副食一：
              - 副食二：
              - 副食三：
              """
    
    guidance = """CKD Stage 1-2 (eGPR >= 60 ml/min/1.73m^2)
                建議攝取量: 不限制鉀的攝取，除非血清鉀濃度升高。
                建議: KDOQI 2004建議每日至少4000毫克。患者若無高血鉀風險，應維持正常鉀攝取量。

                CKD Stage 3-4 (eGPR 30-59 ml/min/1.73m^2)
                建議攝取量: 每日鉀攝取量2000-4000毫克。
                建議: 對於CKD患者增加水果和蔬菜的攝取以減少體重、血壓和淨酸生成（KDOQI 2020），有助於延緩CKD進程。

                CKD Stage 3b-5 (非透析病患，eGPR < 45 ml/min/1.73m^2)
                建議攝取量: 每日鉀攝取量2000-3000毫克。
                建議: 若血清鉀水平超過5 mmol/L，則建議控制鉀含量高的食物，並建議採用浸泡或煮沸等食材處理方式以降低鉀含量。

                CKD Stage 5 (透析患者，eGPR < 15 ml/min/1.73m^2)
                透析患者建議攝取量:
                血液透析（HD）: 每日鉀攝取量2000-3000毫克。
                腹膜透析（PD）: 每日鉀攝取量2000-4000毫克。
                建議: 對透析患者使用低鉀透析液、延長透析時間、增加透析頻率等方式來改善鉀的去除效果。

                特別說明
                無鉀攝取指導共識: 目前對於CKD患者低鉀飲食的建議缺乏一致的指導。不同組織的建議有所不同，且缺乏強而有力的隨機對照試驗證據來支持嚴格的鉀攝取限制。
                漸進、個別化調整: 推薦進行個別化、漸進的鉀攝取減少，以確保CKD患者在避免高血鉀風險的同時，仍能攝取足量的新鮮水果與蔬菜。
                """

    ch_prompt_v3 =  f"""
              你現在扮演一位專業營養師，我是一位腎臟病患者。請用正體中文回答，內容不超過200字元。

              我今天的抽血結果為：
              - 鉀 (K): {health_data[1]} mmol/L
              - 磷 (P): {health_data[2]} mg/dL
              - 隨機血糖 (Glucose random): {health_data[3]} mg/dL
              - 空腹血糖 (Glucose AC): {health_data[4]} mg/dL
              - 低密度脂蛋白膽固醇 (LDL-C): {health_data[5]} mg/dL
              - 腎絲球過濾率 (eGPR): {health_data[6]} ml/min/1.73m^2

              抽血結果建議正常值為：
              - 鉀 (K): 3.4-4.7 mmol/L
              - 磷 (P): 3.5-5.5 mg/dL
              - 隨機血糖 (Glucose random): 小於 125 mg/dL
              - 空腹血糖 (Glucose AC): 70-140 mg/dL
              - 低密度脂蛋白膽固醇 (LDL-C): 小於 130 mg/dL
              - 腎絲球過濾率 (eGPR): 大於 90 ml/min/1.73m^2

              我今天的飲食內容：{food_prompt}。
              
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

              最後，請根據我的抽血結果和今天的飲食內容，提供幾點條列式的飲食建議。此外，根據每日所需營養素以及你所學的菜單知識為我設計下一餐（每個類別只選擇一道菜，不需解釋）：

              菜單：
              - 主食：
              - 副食一：
              - 副食二：
              - 副食三：
              """
    client = Client(host='http://localhost:11434')
    if healthy:
        response = client.chat(model='llama3.1:8b-instruct-q4_K_S', messages=[       #current model list: llama2, llama3.1:70b, llama3.2, llama3.1:70b-instruct-q4_K_S, llama3.1:8b-instruct-fp16, llama3.2:3b-instruct-fp16
        {
            'role': 'user',
            'content': healthy_prompt
        },
        ],
    )
    else:
        response = client.chat(model='llama3.1:8b-instruct-q4_K_S', messages=[       #current model list: llama2, llama3.1:70b, llama3.2, llama3.1:70b-instruct-q4_K_S, llama3.1:8b-instruct-fp16, llama3.2:3b-instruct-fp16
        {
            'role': 'user',
            'content': ch_prompt_v2
        },
        ],
    )
    
    nutrition_data_json_pie = {
        "碳水化合物(g)": nutrition_data[3],    # 單位: g
        "蛋白質(g)": nutrition_data[4],        # 單位: g
        "脂肪(g)": nutrition_data[5],          # 單位: g
    }

    nutrition_data_json_bar_0 = {
        "熱量(kcal)": nutrition_data[2],          # 單位: kcal
        "鉀(mg)": nutrition_data[6],            # 單位: mg
        "磷(mg)": nutrition_data[7],            # 單位: mg
        "鈉(mg)": nutrition_data[12],           # 單位: mg
        "鈣(mg)": nutrition_data[13],           # 單位: mg
    }
    nutrition_data_json_bar_1 = {
        "維生素D(µg)": nutrition_data[9],       # 單位: µg
        "維生素E(mg)": nutrition_data[10],      # 單位: mg
        "維生素K(µg)": nutrition_data[11]       # 單位: µg
    }
    recommand_data_json_bar = {
        "熱量(kcal)": int(recommand_intake[0]),
        "鉀(mg)": 0,            # 單位: mg
        "磷(mg)": 700,            # 單位: mg
        "鈉(mg)": 2400,           # 單位: mg
        "鈣(mg)": int(recommand_intake[4]),
    }
    recommand_data_json_bar_1 = {
        "維生素D(µg)": 15,       # 單位: µg
        "維生素E(mg)": 15,      # 單位: mg
        "維生素K(µg)": 100,       # 單位: µg
    }
    #nutrition_data = nutrition_intake(food_label, email = 'alice@example.com', password = 'securepassword123', date = today_str)
    advice_save_path = os.path.join(save_dir, 'llm_advice.txt')
    json_pie_path = os.path.join(save_dir, "pie.json")
    json_bar_path_0 = os.path.join(save_dir, "bar_0.json")
    json_bar_path_1 = os.path.join(save_dir, "bar_1.json")
    json_recommand_bar_path = os.path.join(save_dir, "recommand.json")
    with open(advice_save_path, 'w') as text_file:
        text_file.write(response['message']['content'])
    with open(json_pie_path, 'w', encoding='utf-8') as f:
        json.dump(nutrition_data_json_pie, f, ensure_ascii=False, indent=4)
    with open(json_bar_path_0, 'w', encoding='utf-8') as f:
        json.dump(nutrition_data_json_bar_0, f, ensure_ascii=False, indent=4)
    with open(json_bar_path_1, 'w', encoding='utf-8') as f:
        json.dump(nutrition_data_json_bar_1, f, ensure_ascii=False, indent=4)
    with open(json_recommand_bar_path, 'w', encoding='utf-8') as f:
        json.dump(recommand_data_json_bar, f, ensure_ascii=False, indent=4)

def nutrition_intake(food_label, email, label_pixel_count, date):
    data = get_user_nutrition_intake(email, date)
    id = find_user_id(email)
    if data is None:
        data = (id[0], date, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    data = list(data)
    df = pd.read_csv(df_path)
    df.set_index('label', inplace=True)             #set label as the index column

    for l in food_label:
        data[2] += df.loc[l, 'calories(kcal)']
        data[3] += df.loc[l, 'carbohydrate(g)']
        data[4] += df.loc[l, 'protein(g)']
        data[5] += df.loc[l, 'fat(g)']
        data[6] += df.loc[l, 'K(mg)']
        data[7] += df.loc[l, 'P(mg)']
        data[8] += df.loc[l, 'VitA(µg)']
        data[9] += df.loc[l, 'VitD(µg)']
        data[10] += df.loc[l, 'VitE(mg)']
        data[11] += df.loc[l, 'VitK(µg)']
        data[12] += df.loc[l, 'Na(mg)']
        data[13] += df.loc[l, 'Ca(mg)']
        print(f'{l}: {df.loc[l]}')

        #print(f"key:{key}, sum:{total}, supp_mass:{supposed_mass}, pom:{portion_of_mass} cal:{df.loc[key, 'calories(kcal)'] * portion_of_mass}")
    for i in range(2,14):
       data[i] = float(data[i])
       data[i] = round(data[i], 2)                  #sometime has weird value like 0.100000000001

    insert_or_update_nutrition(data)

def get_nutrition_intake(email):
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    data = get_user_nutrition_intake(email, today_str)
    id = find_user_id(email)
    if data is None:
        data = (id[0], today_str, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    data = list(data)
    return data

    
      


   