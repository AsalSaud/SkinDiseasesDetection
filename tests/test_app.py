import pytest
import os
from pytest_mock import mocker
from app import app, db, Member, Image
from flask import  session
from unittest.mock import patch, Mock
from werkzeug.security import generate_password_hash
from datetime import datetime
from faker import Faker
from flask import template_rendered


fake = Faker()

class MockFileStorage:
    def __init__(self, filename):
        self.filename = filename

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.test_client() as client:
        with app.app_context():
            db.create_all()
        yield client

@pytest.fixture
def mock_request(monkeypatch):
    with app.test_request_context():
        def mock_files_get(key):
            if key == 'file':
                return MockFileStorage('test_image.jpg')
            return None
        monkeypatch.setattr('flask.request.files.get', mock_files_get)
#-------------------PASS---------------------
def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'<title>Skin Disease Detection</title>' in response.data

#-------------------PASS---------------------
def test_detect_route_no_file_part(client):
    response = client.post('/detect')
    assert response.status_code == 400
    assert response.json == {'error': 'No file part'}

#-------------------PASS----------------------
def test_detect_route_no_selected_file(client):
    data = {'file': ''}
    response = client.post('/detect', data=data)
    assert response.status_code == 400
    assert response.json == {'error': 'No file part'}
#-------------------PASS----------------------
def test_detect_route_success(client):
    image_path = os.path.join('tests', 'image.jpg')
    with open(image_path, 'rb') as f:
        data = {'file': (f, 'test_image.jpg')}

        response = client.post('/detect', data=data, content_type='multipart/form-data')
        assert response.status_code == 200

        json_data = response.json
        assert 'result' in json_data
        assert 'accuracy' in json_data

        assert isinstance(json_data['result'], str)
        assert isinstance(json_data['accuracy'], str)
#------------------PASS-----------------------
def test_get_signup_route(client):
    response = client.get('/signup')
    assert response.status_code == 200
#------------------PASS-----------------------
def test_post_signup_route_empty_fields(client): #  tests the behavior of the /signup route when the user submits the signup form with empty fields.
    response = client.post('/signup', data=dict(username='', email='', dob='', sex='', password='', confirmPassword=''), follow_redirects= True)
    assert response.status_code == 200
    assert b'All fields are required' in response.data
#------------------PASS-----------------------
def test_post_signup_route_passwords_not_matching(client):
    response = client.post('/signup', data=dict(username='test', email='test@example.com', dob='1990-01-01', sex='male', password='password', confirmPassword='differentpassword'), follow_redirects= True)
    assert response.status_code == 200
    assert b'Passwords do not match' in response.data
#------------------PASS------------------------
def test_post_signup_route_success(client):
    with app.app_context():
        date_of_birth = datetime.strptime('1990-01-01', '%Y-%m-%d').date()

        user = Member(username='username14', email = 'mytest13@example.com', date_of_birth = date_of_birth, sex= 'female', password='password')
        db.session.add(user)
        db.session.commit()

        response = client.post('/signup', data=dict(
            username='username14',
            email= 'mytest13@example.com',
            date_of_birth= date_of_birth, 
            sex='female',
            password='password'
        ), follow_redirects=True)
        
        assert response.status_code == 200
        assert b"Login" in response.data
#-------------------PASS--------------------
def test_login_get(client):
    response = client.get('/login')
    assert response.status_code == 200
    assert b"Login" in response.data
#---------------PASS---------------------  
def test_successful_login(client):
    with app.app_context():
        dummy_email = 'test10@example.com'
        dummy_DOB= datetime.strptime('1990-01-01', '%Y-%m-%d').date()
        dummy_sex = 'male'
        
        user = Member(username='testuser10', email = dummy_email, date_of_birth = dummy_DOB, sex = dummy_sex, password=generate_password_hash('testpass'))
        db.session.add(user)
        db.session.commit()

        response = client.post('/login', data=dict(
            username='testuser10',
            password='testpass'
        ), follow_redirects=True)
        assert response.status_code == 200
        assert b"Welcome to our Skin Disease Detection app" in response.data
#-------------------PASS---------------------
def test_login_wrong_password(client):
    with app.app_context():
        dummy_email = 'test2@example.com'
        dummy_date_of_birth = datetime.strptime('1990-01-01', '%Y-%m-%d').date()
        dummy_sex = 'male'

        user = Member(username='wrongname2', email = dummy_email, date_of_birth = dummy_date_of_birth, sex= dummy_sex, password='wrongpaass')
        db.session.add(user)
        db.session.commit()

        response = client.post('/login', data=dict(
            username='wrongname2',
            email=dummy_email,
            date_of_birth=dummy_date_of_birth, 
            sex=dummy_sex,
            password='wrongpass'
        ), follow_redirects=True)
        assert b"Invalid username or password" in response.data
# ------------------------PASS-----------------------
def test_login_nonexistent_user(client):
    response = client.post('/login', data={'username': 'nonexistent', 'password': 'nopass'})
    assert b"User does not exist" in response.data
#-------------------------PASS-------------------------
def test_view_history(client):
    with client.session_transaction() as sess:
        sess['user_id'] = 1 

    response = client.get('/view_history')
    assert response.status_code == 200
    assert b'History' in response.data 

