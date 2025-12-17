#!/usr/bin/env python3
"""
Script tạo user mới hoặc update password cho user
Password sẽ được hash trước khi lưu vào database
"""

import psycopg2
from werkzeug.security import generate_password_hash
import sys
import os
import getpass
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


def add_single_employee_interactive():
	"""Interactive mode để thêm một nhân viên mới"""
	print("=" * 60)
	print("         THÊM NHÂN VIÊN MỚI (Interactive Mode)")
	print("=" * 60)
	
	# Input thông tin
	emp_code = input("Mã nhân viên (VD: NV01): ").strip()
	if not emp_code:
		raise ValueError("Mã nhân viên không được để trống")
	
	full_name = input("Họ và tên đầy đủ (VD: Nguyễn Văn A): ").strip()
	if not full_name:
		raise ValueError("Họ tên không được để trống")
	
	username = input(f"Username đăng nhập [mặc định: {emp_code}]: ").strip() or emp_code
	
	password = getpass.getpass("Mật khẩu (ẩn khi nhập): ").strip()
	if not password:
		raise ValueError("Mật khẩu không được để trống")
	password_confirm = getpass.getpass("Xác nhận mật khẩu: ").strip()
	if password != password_confirm:
		raise ValueError("Mật khẩu xác nhận không khớp!")
	
	print("\nChọn Role:")
	print("  1. admin  - Quản trị viên (full access)")
	print("  2. root   - Super admin")
	print("  3. user   - Nhân viên thường")
	role_choice = input("Nhập số [1/2/3, mặc định: 3]: ").strip() or "3"
	role_map = {"1": "admin", "2": "root", "3": "user"}
	role = role_map.get(role_choice, "user")
	
	print("\nChọn giới tính:")
	print("  M - Nam")
	print("  F - Nữ")
	print("  O - Khác")
	gender = input("Nhập giới tính [M/F/O, để trống để bỏ qua]: ").strip().upper() or None
	if gender and gender not in ['M', 'F', 'O']:
		gender = None
	
	dob = input("Ngày sinh (YYYY-MM-DD, để trống để bỏ qua): ").strip() or None
	
	print("\n" + "-" * 60)
	print("THÔNG TIN SẼ TẠO:")
	print("-" * 60)
	print(f"Mã NV:      {emp_code}")
	print(f"Họ tên:     {full_name}")
	print(f"Username:   {username}")
	print(f"Password:   {'*' * len(password)}")
	print(f"Role:       {role}")
	print(f"Giới tính:  {gender or 'N/A'}")
	print(f"Ngày sinh:  {dob or 'N/A'}")
	print("-" * 60)
	
	confirm = input("\nXác nhận tạo nhân viên này? [y/N]: ").strip().lower()
	if confirm != 'y':
		print("❌ Đã hủy")
		return False
	
	# Tạo employee và login
	create_employee_and_login(
		emp_code=emp_code,
		full_name=full_name,
		username=username,
		password=password,
		role=role,
		gender=gender,
		date_of_birth=dob,
		status='active'
	)
	
	print(f"\n✅ Đã tạo thành công nhân viên {emp_code} ({full_name})")
	print(f"   Username: {username}")
	print(f"   Role: {role}")
	return True


def _cli_args_help():
	return (
		"Usage:\n"
		"  1. Interactive mode (khuyên dùng):\n"
		"     python create_user_login.py add\n\n"
		"  2. Single create:\n"
		"     python create_user_login.py emp_code full_name username password role [gender] [dob]\n"
		"     Example: python create_user_login.py NV10 'Nguyen Van A' NV10 pass123 user M 1990-05-01\n\n"
		"  3. Bulk import:\n"
		"     python create_user_login.py import_dir <dir_path> <default_password> [role]\n"
	)


if __name__ == '__main__':
	# CLI modes:
	# 1) interactive add: create_user_login.py add (hoặc không có args)
	# 2) single create: create_user_login.py emp_code full_name username password role [gender] [dob]
	# 3) bulk import: create_user_login.py import_dir <dir_path> <default_password> [role]
	
	# Show help
	if len(sys.argv) >= 2 and sys.argv[1] in ['--help', '-h', 'help']:
		print(_cli_args_help())
		sys.exit(0)
	
	# Mode 1: Interactive add (khi chạy không có args hoặc với 'add')
	if len(sys.argv) == 1 or (len(sys.argv) >= 2 and sys.argv[1] == 'add'):
		try:
			add_single_employee_interactive()
			sys.exit(0)
		except KeyboardInterrupt:
			print("\n\n❌ Đã hủy bởi người dùng")
			sys.exit(1)
		except Exception as e:
			print(f"\n❌ Lỗi: {e}")
			import traceback
			traceback.print_exc()
			sys.exit(2)
	
	# Mode 2: Bulk import
	if len(sys.argv) >= 2 and sys.argv[1] == 'import_dir':
		if len(sys.argv) < 4:
			print("Usage: create_user_login.py import_dir <dir_path> <default_password> [role]")
			sys.exit(1)
		dir_path = sys.argv[2]
		default_password = sys.argv[3]
		default_role = sys.argv[4] if len(sys.argv) > 4 else 'user'

		def import_from_data_face_dir(path, default_password, default_role='user'):
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

	# Mode 3: fallback - single create
	if len(sys.argv) < 6:
		print(_cli_args_help())
		print("\n💡 Tip: Dùng 'python create_user_login.py add' để thêm nhân viên dễ dàng hơn!")
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



