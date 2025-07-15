import sqlite3
import numpy as np
import os
from datetime import datetime
import bcrypt

db_path = os.path.abspath('FoodSAM/FoodSAM_tools/test.db')

password = 'securepassword123'

def get_recommand_intake(email):
    id = find_user_id(email)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT users.sex, users.age, users.height, users.weight, users.exercise
        FROM users
        WHERE users.id = ? 
        LIMIT 1
    """
    cursor.execute(query, (id[0], ))

    data = cursor.fetchone()
    BMR = 0

    if(data[0] == 1):
        BMR = int(655.1 + (9.563 * data[3]) + (1.850 * data[2]) - (4.676 * data[1]))
    else:
        BMR = int(66.5 + (13.75 * data[3]) + (5.003 * data[2]) - (6.75 * data[1]))

    if(data[4] == 0):
        BMR *= 1.2
    elif(data[4] == 1 or data[4] == 2):
        BMR *= 1.375
    elif(data[4] == 3 or data[4] ==4):
        BMR *= 1.55
    elif(data[4] == 5 or data[4] ==6):
        BMR *= 1.725
    else:
        BMR *= 1.9
    Ca_intake = 0
    if(data[1] <= 9):
        Ca_intake = 800
    elif(data[1] <= 12 and data[1] >= 10):
        Ca_intake = 1000
    elif(data[1] <= 13 and data[1] >= 18):
        Ca_intake = 1200
    else:
        Ca_intake = 1000
        
    carbohydrate = int(BMR * 0.5 / 4)
    fat = int(BMR * 0.3 / 9)
    protein = int(data[3])
    # Close the connection
    conn.close()

    return (BMR, carbohydrate, protein, fat, Ca_intake)

def get_user_nutrition_intake(email, date):
    id = find_user_id(email)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT daily_nutrition.user_id, daily_nutrition.datetime, daily_nutrition.calories, daily_nutrition.carbohydrate, daily_nutrition.protein, daily_nutrition.fat, daily_nutrition.K, daily_nutrition.P, daily_nutrition.VitA, daily_nutrition.VitD, daily_nutrition.VitE, daily_nutrition.VitK, daily_nutrition.Na, daily_nutrition.Ca
        FROM users
        JOIN daily_nutrition ON users.id = daily_nutrition.user_id
        WHERE users.id = ? AND daily_nutrition.datetime = ? 
        LIMIT 1
    """
    cursor.execute(query, (id[0], date))

    # Fetch the single most recent row (user_name, K, P, Glucose_random, Glucose_AC, LDL_C, datetime)
    data = cursor.fetchone()
    
    # Close the connection
    conn.close()

    return data

def get_newest_health_data(email):
    id = find_user_id(email)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(id)
    query = """
        SELECT users.name, health_status.K, health_status.P, health_status.Glucose_random, health_status.Glucose_AC, health_status.LDL_C, health_status.eGPR, health_status.datetime
        FROM users
        JOIN health_status ON users.id = health_status.user_id
        WHERE users.id = ?
        ORDER BY datetime DESC 
        LIMIT 1
    """
    cursor.execute(query, (id[0],))

    # Fetch the single most recent row (user_name, K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime)
    data = cursor.fetchone()

    # Close the connection
    conn.close()

    return data

def predict_mask_with_data(unique_labels, email):
    healthy = True
    out_of_range_count = 0
    color_list_path = 'FoodSAM/FoodSAM_tools/color_list_v2.npy'
    mask_dict = {}
    not_to_eat = [2,4,9,12,15,17,18,19,21,22,23,24,47,49,50,53,54,55,56,59,61,68,70,97,98,99,100,101]

    data = get_newest_health_data(email)
    '''
    K: 3.4-4.7mmol/L
    P: 3.5-5.5 mg/dL
    Glucose random: <125 mg/dL
    Glucose AC: 70-140 mg/dL
    LDL-C: x<100(optimal) 100<x<129(near optimal) 130<x<159(borderline high) 160<x<189(high) x>190(very high) mg/dL
        -> <130 mg/dL
    '''
    print(data)
    #if(data[1] > 4.7 or data[2] > 5.5 or data[3] > 125 or data[4] > 140 or data[5] > 160):
    #    healthy = False
    if data is None:
        data = ['Null', 4.5, 4.5, 120, 90, 100, 100]
    if(data[1] > 4.7):
        out_of_range_count = out_of_range_count + 1
    if(data[2] > 5.5):
        out_of_range_count = out_of_range_count + 1
    if(data[3] > 125):
        out_of_range_count = out_of_range_count + 1
    if(data[4] > 130):
        out_of_range_count = out_of_range_count + 1
    if(data[5] > 160):
        out_of_range_count = out_of_range_count + 1
        
    healthy = False

    color_list = np.load(color_list_path)    
    print(out_of_range_count)           
    
    if(out_of_range_count <= 1):
        color_mask = np.array([50,230,255])      #yellow
    else:
        color_mask = np.array([0,0,255])        #red

    if healthy:
        for i in unique_labels:
            mask_dict[i] = color_list[i]
    else:
        for i in unique_labels:
            if i in not_to_eat:
                mask_dict[i] = color_mask
            else:
                mask_dict[i] = color_list[i]
            

    return mask_dict, healthy, out_of_range_count


def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def add_user(name, email, age, password, height, weight, sex, exercise):
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    hashed_password = hash_password(password)
    if sex == 'female':
        sex_tag = int(1)
    else:
        sex_tag = int(0)
    height = float(height)
    weight = float(weight)
    cursor.execute("INSERT INTO users (name, email, age, password, height, weight, sex, exercise) VALUES (?, ?, ?, ? ,? ,?, ?, ?)", (name, email, age, hashed_password, height, weight, sex_tag, exercise))
    conn.commit()
    conn.close()

def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

def login_user(email, password):
    correct = False
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    if result and verify_password(result[0], password):
        print("Login successful")
        correct = True
    else:
        print("Invalid credentials")
    conn.commit()
    conn.close()
    return correct

def find_user_id(email):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email, ))
    conn.commit()
    return cursor.fetchone()  # This will return the first match or None if not found

def get_user_health_data(user_id):
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute('''
    SELECT users.name, users.email, users.age, health_status.K, health_status.P, health_status.Glucose_random, health_status.Glucose_AC, health_status.LDL_C, health_status.eGPR, health_status.datetime
    FROM users
    JOIN health_status ON users.id = health_status.user_id
    WHERE users.id = ?
    ''', (user_id,))
    conn.commit()
    return cursor.fetchall()


def find_user_data(email, password):
    id = login_and_get_id(email, password)
    user_health_data = get_user_health_data(id)
    for data in user_health_data:
        print(data)

def add_health_status(email, K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime):
    id = find_user_id(email)
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO health_status (user_id, K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (id[0], K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime))
    conn.commit()
    conn.close()

def login_and_get_id(email, password):
    correct = login_user(email, password)
    if not correct:
        return
    id = find_user_id(email)
    return id[0]

def insert_or_update_nutrition(data):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    #print(data)
    #print([type(value) for value in data])
    # SQL query to insert or update the daily nutrition data
    query = """
        INSERT INTO daily_nutrition (user_id, datetime, calories, carbohydrate, protein, fat, K, P, VitA, VitD, VitE, VitK, Na, Ca)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, datetime)
        DO UPDATE SET 
            calories = excluded.calories,
            carbohydrate = excluded.carbohydrate,
            protein = excluded.protein,
            fat = excluded.fat,
            K = excluded.K,
            P = excluded.P,
            VitA = excluded.VitA,
            VitD = excluded.VitD,
            VitE = excluded.VitE,
            VitK = excluded.VitK,
            Na = excluded.Na,
            Ca = excluded.Ca
    """

    # Execute the query with provided values
    cursor.execute(query, (data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13]))
    
    # Commit the transaction and close the connection
    conn.commit()
    conn.close()
