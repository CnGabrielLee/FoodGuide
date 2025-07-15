import sqlite3
import argparse
import sys
import bcrypt

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def add_user(name, email, age, password, height, weight, sex, exercise):
    conn = None
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('FoodSAM_tools/test.db')
        cursor = conn.cursor()

        # Hash the password
        hashed_password = hash_password(password)

        # Convert sex to integer
        sex_tag = int(sex)

        # Convert numeric fields
        age = int(age)
        height = float(height)
        weight = float(weight)
        exercise = int(exercise)
        
        # Insert user data
        cursor.execute("""
            INSERT INTO users 
            (name, email, age, password, height, weight, sex, exercise) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, email, age, hashed_password, height, weight, sex_tag, exercise))
        
        conn.commit()
        print("User added successfully")
    except Exception as e:
        print(f"Error adding user: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description='Add user to database')
    parser.add_argument('--name', required=True, help='User name')
    parser.add_argument('--email', required=True, help='User email')
    parser.add_argument('--age', required=True, help='User age')
    parser.add_argument('--password', required=True, help='User password')
    parser.add_argument('--height', required=True, help='User height')
    parser.add_argument('--weight', required=True, help='User weight')
    parser.add_argument('--sex', required=True, help='User sex (male/female)')
    parser.add_argument('--exercise', required=True, help='Exercise level')

    args = parser.parse_args()

    add_user(
        args.name, 
        args.email, 
        args.age, 
        args.password, 
        args.height, 
        args.weight, 
        args.sex, 
        args.exercise
    )

if __name__ == '__main__':
    main()