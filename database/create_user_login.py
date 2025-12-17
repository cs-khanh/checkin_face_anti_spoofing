#!/usr/bin/env python3
"""
Script tạo user mới hoặc update password cho user
Password sẽ được hash trước khi lưu vào database
"""

import psycopg2
from werkzeug.security import generate_password_hash
import sys
import os
from connectDB import get_db_connection



def create_employee_and_login(emp_code, full_name, username, password, role,
							  gender=None, date_of_birth=None, status='active', face_file_uri=None):
	"""Create an employee row and corresponding login row in a single transaction.

	Returns True on success, raises Exception on failure.
	"""
	pool = get_db_connection()
	if not pool:
		raise RuntimeError("Database pool not available")

	conn = None
	try:
		conn = pool.getconn()
		# Ensure we run in a transaction
		conn.autocommit = False
		cur = conn.cursor()

		# Basic existence checks
		cur.execute("SELECT 1 FROM employees WHERE emp_code = %s", (emp_code,))
		if cur.fetchone():
			raise ValueError(f"Employee with emp_code '{emp_code}' already exists")

		cur.execute("SELECT 1 FROM login WHERE username = %s", (username,))
		if cur.fetchone():
			raise ValueError(f"Login with username '{username}' already exists")

		# Insert employee
		insert_emp_sql = (
			"INSERT INTO employees (emp_code, full_name, gender, date_of_birth, status, face_file_uri)"
			" VALUES (%s, %s, %s, %s, %s, %s)"
		)
		cur.execute(insert_emp_sql, (emp_code, full_name, gender, date_of_birth, status, face_file_uri))

		# Hash password
		hashed = generate_password_hash(password)

		# Insert login (username constraint is enforced by DB)
		insert_login_sql = (
			"INSERT INTO login (username, password, role, emp_code) VALUES (%s, %s, %s, %s)"
		)
		cur.execute(insert_login_sql, (username, hashed, role, emp_code))

		# commit transaction
		conn.commit()
		cur.close()
		return True
	except Exception:
		if conn:
			conn.rollback()
		raise
	finally:
		if conn:
			pool.putconn(conn)


def _cli_args_help():
	return ("Usage: create_user_login.py emp_code full_name username password role [gender] [date_of_birth YYYY-MM-DD]\n"
			"Example: create_user_login.py NV10 'Nguyen Van A' NV10 pass123 staff M 1990-05-01")


if __name__ == '__main__':
	# CLI modes:
	# 1) single create: create_user_login.py emp_code full_name username password role [gender] [dob]
	# 2) bulk import: create_user_login.py import_dir <dir_path> <default_password> [role]
	if len(sys.argv) >= 2 and sys.argv[1] == 'import_dir':
		if len(sys.argv) < 4:
			print("Usage: create_user_login.py import_dir <dir_path> <default_password> [role]")
			sys.exit(1)
		dir_path = sys.argv[2]
		default_password = sys.argv[3]
		default_role = sys.argv[4] if len(sys.argv) > 4 else 'staff'

		def import_from_data_face_dir(path, default_password, default_role='staff'):
			successes = []
			failures = []
			if not os.path.isdir(path):
				raise ValueError(f"Path not found or not a directory: {path}")
			for name in sorted(os.listdir(path)):
				full_path = os.path.join(path, name)
				if not os.path.isdir(full_path):
					continue
				# Expect names like NV01_something
				if '_' not in name:
					failures.append((name, 'invalid folder name format'))
					continue
				emp_code, raw = name.split('_', 1)
				# Derive a readable full name: replace underscores with spaces and title-case
				full_name = raw.replace('_', ' ').replace('-', ' ').title()
				username = emp_code
				try:
					create_employee_and_login(emp_code, full_name, username, default_password, default_role,
											  face_file_uri=full_path)
					successes.append((emp_code, username))
				except Exception as e:
					failures.append((name, str(e)))
			return successes, failures

		try:
			s, f = import_from_data_face_dir(dir_path, default_password, default_role)
			print(f"✅ Imported {len(s)} entries")
			for emp, user in s:
				print(f"  - {emp} -> {user}")
			if f:
				print(f"⚠️ {len(f)} failures:")
				for name, reason in f:
					print(f"  - {name}: {reason}")
				sys.exit(2)
			sys.exit(0)
		except Exception as e:
			print(f"❌ Import failed: {e}")
			sys.exit(3)

	# fallback: single create
	if len(sys.argv) < 6:
		print(_cli_args_help())
		sys.exit(1)

	emp_code = sys.argv[1]
	full_name = sys.argv[2]
	username = sys.argv[3]
	password = sys.argv[4]
	role = sys.argv[5]
	gender = sys.argv[6] if len(sys.argv) > 6 else None
	dob = sys.argv[7] if len(sys.argv) > 7 else None

	try:
		ok = create_employee_and_login(emp_code, full_name, username, password, role, gender, dob)
		if ok:
			print(f"✅ Created employee {emp_code} and login {username}")
	except Exception as e:
		print(f"❌ Failed: {e}")
		sys.exit(2)



