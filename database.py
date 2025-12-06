import sqlite3
import hashlib
import json
from datetime import datetime

class Database:
    def __init__(self, db_name='health.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.init_database()
    
    def get_connection(self):
        return self.conn
    
    def init_database(self):
        """Create tables if not exists"""
        cursor = self.conn.cursor()
        
        # Users table with profile_photo
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                profile_photo LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Add profile_photo column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN profile_photo LONGTEXT')
        except:
            pass
        
        # Predictions table with ALL fields (diet, workout included)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                disease TEXT,
                confidence REAL,
                model_used TEXT,
                symptoms TEXT,
                description TEXT,
                medications TEXT,
                precautions TEXT,
                diet TEXT,
                workout TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Add diet column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE predictions ADD COLUMN diet TEXT')
        except:
            pass
        
        # Add workout column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE predictions ADD COLUMN workout TEXT')
        except:
            pass
        
        self.conn.commit()
        print("✓ Database initialized with all columns (diet, workout, profile_photo)")
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, name, email, password):
        """Register new user"""
        try:
            cursor = self.conn.cursor()
            hashed_password = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (name, email, password)
                VALUES (?, ?, ?)
            ''', (name, email, hashed_password))
            
            user_id = cursor.lastrowid
            self.conn.commit()
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': user_id
            }
            
        except sqlite3.IntegrityError:
            return {
                'success': False,
                'message': 'Email already exists'
            }
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def login_user(self, email, password):
        """Verify user credentials"""
        try:
            cursor = self.conn.cursor()
            hashed_password = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, name, email, created_at FROM users
                WHERE email = ? AND password = ?
            ''', (email, hashed_password))
            
            user = cursor.fetchone()
            
            if user:
                return {
                    'success': True,
                    'message': 'Login successful',
                    'user': {
                        'id': user[0],
                        'name': user[1],
                        'email': user[2],
                        'created_at': user[3]
                    }
                }
            else:
                return {
                    'success': False,
                    'message': 'Invalid email or password'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': str(e)
            }
    
    def save_prediction(self, user_id, disease, confidence, model_used, symptoms, description=None, medications=None, precautions=None, diet=None, workout=None):
        """Save prediction to history with diet & workout"""
        try:
            cursor = self.conn.cursor()
            
            # Convert lists to JSON strings
            symptoms_json = json.dumps(symptoms) if symptoms else '[]'
            medications_json = json.dumps(medications) if medications else '[]'
            precautions_json = json.dumps(precautions) if precautions else '[]'
            diet_json = json.dumps(diet) if diet else '[]'
            workout_json = json.dumps(workout) if workout else '[]'
            
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
            cursor.execute('''
                INSERT INTO predictions 
                (user_id, disease, confidence, model_used, symptoms, description, medications, precautions, diet, workout, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, disease, confidence, model_used, symptoms_json, description, medications_json, precautions_json, diet_json, workout_json, timestamp))
            
            self.conn.commit()
            return {'success': True, 'message': 'Prediction saved'}
        
        except Exception as e:
            self.conn.rollback()
            print(f"Error saving prediction: {e}")
            return {'success': False, 'message': str(e)}
    
    def get_user_history(self, user_id):
        """Get prediction history for user with ALL fields"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT id, disease, confidence, model_used, symptoms, description, medications, precautions, diet, workout, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            history = cursor.fetchall()
            
            result = []
            for row in history:
                result.append({
                    'id': row[0],
                    'disease': row[1],
                    'confidence': row[2],
                    'model_used': row[3],
                    'symptoms': json.loads(row[4]) if row[4] else [],
                    'description': row[5],
                    'medications': json.loads(row[6]) if row[6] else [],
                    'precautions': json.loads(row[7]) if row[7] else [],
                    'diet': json.loads(row[8]) if row[8] else [],
                    'workout': json.loads(row[9]) if row[9] else [],
                    'date': row[10]
                })
            
            return result
        
        except Exception as e:
            print(f"Error getting history: {e}")
            return []
    
    def delete_prediction(self, prediction_id):
        """Delete a prediction from history"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
            self.conn.commit()
            return {'success': True}
        
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def update_user_profile(self, user_id, new_name=None, new_email=None, profile_photo=None):
        """Update user profile (name, email, photo)"""
        try:
            cursor = self.conn.cursor()
            
            # Check if email is already taken
            if new_email:
                cursor.execute('''
                    SELECT id FROM users WHERE email = ? AND id != ?
                ''', (new_email, user_id))
                
                if cursor.fetchone():
                    return {'success': False, 'message': 'Email already in use'}
            
            # Build update query
            updates = []
            params = []
            
            if new_name:
                updates.append('name = ?')
                params.append(new_name)
            
            if new_email:
                updates.append('email = ?')
                params.append(new_email)
            
            if profile_photo:
                updates.append('profile_photo = ?')
                params.append(profile_photo)
            
            if not updates:
                return {'success': False, 'message': 'No updates provided'}
            
            params.append(user_id)
            
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            self.conn.commit()
            
            if cursor.rowcount > 0:
                return {'success': True, 'message': 'Profile updated'}
            else:
                return {'success': False, 'message': 'User not found'}
        
        except sqlite3.IntegrityError:
            return {'success': False, 'message': 'Email already exists'}
        except Exception as e:
            self.conn.rollback()
            return {'success': False, 'message': str(e)}
    
    def change_user_password(self, user_id, current_password, new_password):
        """Change user password"""
        try:
            cursor = self.conn.cursor()
            hashed_current = self.hash_password(current_password)
            
            cursor.execute('''
                SELECT id FROM users WHERE id = ? AND password = ?
            ''', (user_id, hashed_current))
            
            if not cursor.fetchone():
                return {'success': False, 'message': 'Current password is incorrect'}
            
            hashed_new = self.hash_password(new_password)
            cursor.execute('UPDATE users SET password = ? WHERE id = ?', (hashed_new, user_id))
            self.conn.commit()
            
            return {'success': True, 'message': 'Password changed'}
        
        except Exception as e:
            self.conn.rollback()
            return {'success': False, 'message': str(e)}
    
    def get_user_by_id(self, user_id):
        """Get user details by ID"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT id, name, email, profile_photo, created_at FROM users
                WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            
            if user:
                return {
                    'success': True,
                    'user': {
                        'id': user[0],
                        'name': user[1],
                        'email': user[2],
                        'profile_photo': user[3],
                        'created_at': user[4]
                    }
                }
            else:
                return {'success': False, 'message': 'User not found'}
        
        except Exception as e:
            return {'success': False, 'message': str(e)}