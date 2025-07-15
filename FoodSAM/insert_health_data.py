import argparse
import sqlite3
from datetime import datetime

def find_user_id(email):
    conn = sqlite3.connect('FoodSAM_tools/test.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email, ))
    conn.commit()
    return cursor.fetchone()  # This will return the first match or None if not found


def add_health_status(email, K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime=None):
    # If no datetime provided, use current time
    if datetime is None:
        datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Find user ID
        id_result = find_user_id(email)
        
        if id_result is None:
            print(f"No user found with email: {email}")
            return False
        
        # Connect to health status database
        conn = sqlite3.connect('FoodSAM_tools/test.db')
        cursor = conn.cursor()
        
        # Insert health status
        cursor.execute(
            "INSERT INTO health_status (user_id, K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (id_result[0], K, P, Glucose_random, Glucose_AC, LDL_C, eGPR, datetime)
        )
        
        conn.commit()
        print("Health status successfully added!")
        return True
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Add health status for a user')
    
    # Add arguments with clear descriptions
    parser.add_argument('--email', type=str, required=True, 
                        help='User email address')
    parser.add_argument('--K', type=float, required=True, 
                        help='Potassium level')
    parser.add_argument('--P', type=float, required=True, 
                        help='Phosphorus level')
    parser.add_argument('--random', type=float, required=True, 
                        help='Random blood glucose level')
    parser.add_argument('--AC', type=float, required=True, 
                        help='Fasting blood glucose level')
    parser.add_argument('--ldl', type=float, required=True, 
                        help='Low-density lipoprotein cholesterol level')
    parser.add_argument('--egfr', type=float, required=True, 
                        help='Estimated glomerular filtration rate')
    parser.add_argument('--datetime', type=str, default=None, 
                        help='Datetime of measurement (optional, defaults to current time)')

    # Parse arguments
    args = parser.parse_args()

    # Call function with parsed arguments
    result = add_health_status(
        email=args.email,
        K=args.K,
        P=args.P,
        Glucose_random=args.random,
        Glucose_AC=args.AC,
        LDL_C=args.ldl,
        eGPR=args.egfr,
        datetime=args.datetime
    )

    # Exit with appropriate status code
    exit(0 if result else 1)

if __name__ == '__main__':
    main()