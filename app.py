import base64
import json
import os
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from flask import jsonify, redirect, url_for
from sqlalchemy.ext.mutable import MutableDict
from flask import render_template
from sqlalchemy.exc import IntegrityError
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pandas as pd
from Detector import *
from create_dataset import capture_and_encode_faces
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')

# Debug mode is now controlled in one place (see app.run at bottom) via env var,
# instead of being set both True and False in two different spots.


# Secret key now comes from the environment. Falls back to a random key for
# local dev only (sessions will invalidate on every restart in that case).
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# The /verifyface liveness check submits one large hidden form field: a
# JSON array of ~20-30 base64-encoded JPEG frames. Recent Werkzeug versions
# default to a 500KB cap on any single non-file form field
# (max_form_memory_size), which that payload comfortably exceeds -- causing
# a 413 "Request Entity Too Large" even though nothing is actually wrong.
# Raise both the per-field and whole-request caps; still bounded, just
# sized for this app's actual payloads instead of Werkzeug's generic
# default.


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024      # 10 MB per request
app.config['MAX_FORM_MEMORY_SIZE'] = 10 * 1024 * 1024    # 10 MB per form field

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)


def utcnow():
    """Single source of truth for 'now' -- always timezone-aware UTC."""
    return datetime.now(timezone.utc)


def to_utc(dt):
    """Make a naive datetime timezone-aware (assumes it was stored as UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def parse_allowed_emails(allowed_emails_str):
    """Split/clean the allowed_emails string into a normalized set of emails."""
    if not allowed_emails_str:
        return set()
    return {e.strip().lower() for e in allowed_emails_str.split(',') if e.strip()}


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    dob = db.Column(db.String(20))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(255))  # widened to fit password hashes
    has_logged_in_before = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)


class Poll(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(255))
    options = db.Column(MutableDict.as_mutable(db.JSON))
    expiry = db.Column(db.DateTime)
    # Store emails separated by comma
    allowed_emails = db.Column(db.String(1000))

    # Define a relationship to the User model
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    admin = db.relationship('User', backref=db.backref('polls', lazy=True))


# Define a new table for vote history
class VoteHistory(db.Model):
    __tablename__ = 'vote_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    poll_id = db.Column(db.Integer, db.ForeignKey('poll.id'))


# Add a relationship to the User model
User.vote_history = db.relationship(
    'VoteHistory', backref='user', lazy='dynamic')

# Add a relationship to the Poll model
Poll.vote_history = db.relationship(
    'VoteHistory', backref='poll', lazy='dynamic')


def render_pie_chart(options, values):
    """Shared pie-chart rendering, used by every route that draws poll results."""
    colors = ['#1b9e77', '#2ca02c', '#43a047', '#66bb6a', '#81c784',
              '#a5d6a7', '#c8e6c9', '#e8f5e9', '#f1f8e9', '#f9fbe7']

    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=options, autopct='%1.1f%%',
            colors=colors, startangle=140, textprops={'fontsize': 25})
    plt.axis('equal')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return img_base64


def format_remaining(expiry):
    """Return HH:MM:SS remaining until expiry (UTC-aware), or '00:00:00' if past."""
    if not expiry:
        return "N/A"
    remaining_seconds = (to_utc(expiry) - utcnow()).total_seconds()
    if remaining_seconds <= 0:
        return "00:00:00"
    hours, remainder = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


@app.route('/thankyou/<poll_id>')
def thankyou_route(poll_id):
    user_id = session.get('user_id')
    if not user_id:
        return "User not found or session expired"

    user = db.session.get(User, user_id)
    if not user:
        return "User not found or session expired"

    poll = db.session.get(Poll, poll_id)
    if not poll:
        return "Poll not found"

    if not poll.options or all(v == 0 for v in poll.options.values()):
        return render_template('ty.html', total_votes=0, user=user,
                               leading_option=None, leading_candidate_votes=0,
                               pie_chart_img=None)

    leading_option = max(poll.options, key=poll.options.get)
    leading_candidate_votes = poll.options.get(leading_option)
    total_votes = sum(poll.options.values())

    pie_chart_img = render_pie_chart(
        list(poll.options.keys()), list(poll.options.values()))

    return render_template('ty.html', total_votes=total_votes, user=user,
                           leading_option=leading_option,
                           leading_candidate_votes=leading_candidate_votes,
                           pie_chart_img=pie_chart_img)


def filter_allowed_polls(user):
    # Filter polls based on user's email (normalized, exact match rather than substring)
    email = (user.email or '').strip().lower()
    return [poll for poll in Poll.query.all() if email in parse_allowed_emails(poll.allowed_emails)]


# Create tables
with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        dob = request.form.get('dob')
        email = request.form.get('email')
        password = request.form.get('password')
        # NOTE: is_admin self-checkbox intentionally left as-is (not in scope of this fix pass)
        is_admin = bool(request.form.get('is_admin'))

        # Check if all fields are filled
        if not (name and dob and email and password):
            return render_template('signup.html', error="Please fill all the fields.")

        try:
            new_user = User(name=name, dob=dob, email=email.strip().lower(),
                            password=generate_password_hash(password),
                            has_logged_in_before=False, is_admin=is_admin)
            db.session.add(new_user)
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return render_template('signup.html', error="Email already exists. Please choose a different email.")

        return redirect('/login')
    return render_template('signup.html')


@app.route('/face', methods=['GET', 'POST'])
def face():
    if request.method == 'POST':
        user_id = session.get('user_id')
        if not user_id:
            return redirect('/login')

        user = db.session.get(User, user_id)
        if not user:
            return "User not found in the database."

        # Images are captured client-side (browser webcam) and sent as a
        # JSON array of base64 data-URLs in a hidden form field. This
        # replaces the old server-side cv2.VideoCapture/imshow loop, which
        # required a display on the server and blocked the Flask worker.
        images_json = request.form.get('image_data')
        if not images_json:
            return render_template('face.html',
                                   error="No images captured. Please allow camera access and try again.")

        try:
            images = json.loads(images_json)
        except (TypeError, ValueError):
            return render_template('face.html', error="Invalid image data received.")

        captured = capture_and_encode_faces(user.email, images)
        if captured == 0:
            return render_template('face.html',
                                   error="No face was detected in the captured images. Please try again with better lighting.")

        return redirect('/dashboard')

    return render_template('face.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            if user.is_admin:
                session['is_admin'] = True
                if not user.has_logged_in_before:
                    user.has_logged_in_before = True
                    db.session.commit()
                    return redirect('/admin/new')
                else:
                    return redirect('/admin')
            elif not user.has_logged_in_before:
                user.has_logged_in_before = True
                db.session.commit()
                return render_template('face.html')
            else:
                return redirect('/dashboard')
        else:
            error = 'Invalid email or password'

    return render_template('login.html', error=error)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'user_id' in session and session.get('is_admin', False):
        admin_id = session['user_id']
        polls = Poll.query.filter_by(admin_id=admin_id).all()
        polls_count = len(polls)
        current_time = utcnow()

        live_polls = [p for p in polls if p.expiry and to_utc(
            p.expiry) > current_time]
        live_polls_count = len(live_polls)

        latest_poll = Poll.query.filter_by(
            admin_id=admin_id).order_by(Poll.id.desc()).first()
        admin_user = User.query.filter_by(id=admin_id).first()

        leading_option = None
        total_votes = 0
        remaining_time = "N/A"
        pie_chart_img = None

        if latest_poll:
            remaining_time = format_remaining(latest_poll.expiry)
            values = list(latest_poll.options.values()
                          ) if latest_poll.options else []

            if values and not all(v == 0 for v in values):
                leading_option = max(latest_poll.options,
                                     key=latest_poll.options.get)
                total_votes = sum(latest_poll.options.values())
                pie_chart_img = render_pie_chart(
                    list(latest_poll.options.keys()), values)

        return render_template('admin.html', polls=polls, polls_count=polls_count,
                               live_polls_count=live_polls_count, latest_poll=latest_poll,
                               admin=admin_user, pie_chart_img=pie_chart_img,
                               leading_option=leading_option, total_votes=total_votes,
                               remaining_time=remaining_time)
    else:
        return redirect('/login')


@app.route('/admin/new', methods=['GET', 'POST'])
def redirect_to_new():
    if 'user_id' in session and session.get('is_admin', False):
        admin = User.query.filter_by(id=session['user_id']).first()
        admin_id = session['user_id']
        polls = Poll.query.filter_by(admin_id=admin_id).all()
        return render_template('create_poll.html', admin=admin, polls=polls)
    else:
        return redirect(url_for('admin'))


def create_options_dict(options_str):
    keys_list = [key.strip() for key in options_str.split(',') if key.strip()]
    return {key: 0 for key in keys_list}


def read_emails_from_file(file_storage):
    """
    Reads a list of emails from an uploaded file. Supports both real CSV files
    and Excel files (the previous version was named 'read_emails_from_csv' but
    only ever called pd.read_excel, so genuine .csv uploads would crash).
    """
    filename = file_storage.filename or ''
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(file_storage)
        else:
            df = pd.read_excel(file_storage)

        emails = df.iloc[:, 0].dropna().astype(
            str).str.strip().str.lower().tolist()
        return ','.join(emails)
    except Exception as e:
        return ""


@app.route('/admin/create_poll', methods=['GET', 'POST'])
def create_poll():
    if request.method == 'POST':
        question = request.form['question']
        options_str = request.form['options']
        expiry_date = request.form['expiry']

        # Allowed emails can come from a manual field and/or an uploaded file;
        # previously this could raise UnboundLocalError if no file was uploaded.
        allowed_emails = request.form.get('allowed_emails', '') or ''
        csv_file = request.files.get('csv_file')
        if csv_file and csv_file.filename:
            file_emails = read_emails_from_file(csv_file)
            allowed_emails = ','.join(
                filter(None, [allowed_emails.strip(), file_emails.strip()])
            )

        options = create_options_dict(options_str)
        # Expiry is entered in the admin's local time via the form; stored as UTC.
        expiry_date = datetime.strptime(
            expiry_date, '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc)

        admin_id = session.get('user_id')

        new_poll = Poll(question=question, options=options, expiry=expiry_date,
                        allowed_emails=allowed_emails, admin_id=admin_id)
        db.session.add(new_poll)
        db.session.commit()

        return redirect('/admin')

    return render_template('create_poll.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    user = db.session.get(User, session['user_id'])
    if not user:
        return "User not found"

    email = (user.email or '').strip().lower()
    current_time = utcnow()

    all_polls = [p for p in Poll.query.all(
    ) if email in parse_allowed_emails(p.allowed_emails)]
    allowed_polls = [p for p in all_polls if p.expiry and to_utc(
        p.expiry) > current_time]

    return render_template('dashboard.html', user=user, allowed_polls=allowed_polls, polls=all_polls)


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    if request.method == 'POST':
        session.clear()
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))


@app.route('/verifyface/<poll_id>', methods=['GET', 'POST'])
def verify(poll_id):
    if request.method != 'POST':
        # The flow starts at /redirect_to_poll/<poll_id>, which is what
        # actually picks a random challenge and stashes it in the session.
        # Anyone landing here directly via GET gets sent there so a
        # challenge exists before we ever try to verify one.
        return redirect(url_for('redirect_to_poll', poll_id=poll_id))

    user_id = session.get('user_id')
    if not user_id:
        return redirect('/login')

    user = db.session.get(User, user_id)
    if not user:
        return redirect('/login')

    # The challenge (blink / turn_left / turn_right) was generated
    # server-side when the verify page was loaded and stored in the
    # session -- we never trust a challenge value coming from the client,
    # since that would let someone simply claim they were given whichever
    # challenge is easiest to spoof.
    challenge = session.get('liveness_challenge')
    challenge_poll_id = session.get('liveness_poll_id')
    if not challenge or challenge_poll_id != str(poll_id):
        return redirect(url_for('redirect_to_poll', poll_id=poll_id))

    challenge_label = CHALLENGE_LABELS[challenge]

    # The browser captures ~2-3 seconds of frames (roughly 20-30) while the
    # user performs the challenge, and sends them all as a JSON array of
    # base64 data-URLs -- same encoding style as the multi-shot /face
    # registration flow, just with many more frames and a liveness check
    # gating the recognition step.
    frames_json = request.form.get('image_data')
    if not frames_json:
        return render_template('verify.html', user=user, poll_id=poll_id,
                               challenge=challenge, challenge_label=challenge_label,
                               error="No images captured. Please allow camera access and try again.")

    try:
        frames = json.loads(frames_json)
    except (TypeError, ValueError):
        return render_template('verify.html', user=user, poll_id=poll_id,
                               challenge=challenge, challenge_label=challenge_label,
                               error="Invalid image data received.")

    success, reason = liveness_verification_loop(user.email, challenge, frames)

    # The challenge is single-use: clear it regardless of outcome so a
    # resubmission of the same captured frames can't be replayed against
    # the same prompt.
    session.pop('liveness_challenge', None)
    session.pop('liveness_poll_id', None)

    if not success:
        if reason == "challenge_failed":
            error = f"Liveness check failed -- we didn't see you \"{challenge_label}\". Please try again."
        elif reason == "no_face":
            error = "We couldn't see your face clearly in enough frames. Please try again with better lighting."
        else:
            error = "Face not recognized."
        return render_template('result_false.html', user=user, poll_id=poll_id, error=error)

    poll = db.session.get(Poll, poll_id)
    if not poll:
        return "Poll not found"

    total_votes = sum(poll.options.values()) if poll.options else 0
    remaining_time = format_remaining(poll.expiry)

    existing_vote = VoteHistory.query.filter_by(
        user_id=user_id, poll_id=poll_id).first()
    if existing_vote:
        return render_template('poll.html', error='User has already voted for this poll',
                               user=user, poll_id=poll_id, poll=poll,
                               remaining_time=remaining_time, total_votes=total_votes)
    else:
        return render_template('poll.html', user=user, poll_id=poll_id, poll=poll,
                               remaining_time=remaining_time, total_votes=total_votes)


@app.route('/submit_vote', methods=['POST'])
def submit_vote():
    user_id = session.get('user_id')
    if not user_id:
        return render_template('result.html', error='User not authenticated', polls=Poll.query.all())

    user = db.session.get(User, user_id)
    poll_id = request.form.get('poll_id')
    selected_option = request.form.get('selected_option')

    poll = db.session.get(Poll, poll_id)
    if not poll:
        return render_template('poll.html', error='Poll not found', user=user, polls=Poll.query.all())

    # Expiry check now uses the same UTC convention as everywhere else.
    if poll.expiry and to_utc(poll.expiry) < utcnow():
        return render_template('poll.html', error='Poll has expired', user=user, polls=Poll.query.all())

    email = (user.email or '').strip().lower()
    if email not in parse_allowed_emails(poll.allowed_emails):
        return render_template('poll.html', error='You are not allowed to vote on this poll',
                               user=user, polls=filter_allowed_polls(user))

    # Reject any option that isn't one of the poll's real choices, to stop
    # arbitrary keys being injected into the results.
    if not poll.options or selected_option not in poll.options:
        return render_template('poll.html', error='Invalid option selected',
                               user=user, poll=poll, poll_id=poll_id)

    existing_vote = VoteHistory.query.filter_by(
        user_id=user_id, poll_id=poll_id).first()
    if existing_vote:
        return render_template('poll.html', error='User has already voted for this poll',
                               user=user, polls=filter_allowed_polls(user))

    poll_options = poll.options
    poll_options[selected_option] = poll_options.get(selected_option, 0) + 1
    poll.options = poll_options

    vote_history = VoteHistory(user_id=user_id, poll_id=poll_id)
    db.session.add(vote_history)
    db.session.commit()

    return redirect(url_for('thankyou_route', poll_id=poll_id, user_id=user_id))


@app.route('/redirect_to_poll/<poll_id>')
def redirect_to_poll(poll_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = db.session.get(User, session['user_id'])
    if not user:
        return redirect('/login')

    # Pick a random liveness challenge for this verification attempt and
    # remember it server-side (tied to this poll_id) so the /verifyface
    # POST can check against it -- the client only ever sees the label to
    # display, never anything the challenge check itself relies on.
    challenge = choose_challenge()
    session['liveness_challenge'] = challenge
    session['liveness_poll_id'] = str(poll_id)

    return render_template('verify.html', user=user, poll_id=poll_id,
                           challenge=challenge, challenge_label=CHALLENGE_LABELS[challenge])


@app.route('/get_result', methods=['POST'])
def submit_form():
    poll_id = request.form.get('poll_id')
    user_id_field = request.form.get('user_id')

    if 'user_id' not in session:
        return redirect('/login')
    user = db.session.get(User, session['user_id'])
    if not user:
        return "User not found"

    poll = db.session.get(Poll, poll_id)
    if not poll:
        return "Poll not found"

    values = list(poll.options.values()) if poll.options else []
    options = list(poll.options.keys()) if poll.options else []
    total_votes = sum(values) if values else 0
    current_time = utcnow()

    remaining_seconds = 0
    if poll.expiry:
        remaining_seconds = (to_utc(poll.expiry) -
                             current_time).total_seconds()
        remaining_seconds = remaining_seconds if remaining_seconds > 0 else 0

    if remaining_seconds > 0:
        days, remainder = divmod(remaining_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        remaining_str = f"{int(days)} days {int(hours)} hr {int(minutes)} min {int(seconds)} sec"
    else:
        remaining_str = "00:00:00"

    email = (user.email or '').strip().lower()
    allowed_polls = [p for p in Poll.query.all()
                     if email in parse_allowed_emails(p.allowed_emails)
                     and p.expiry and to_utc(p.expiry) > current_time]
    all_polls = [p for p in Poll.query.all(
    ) if email in parse_allowed_emails(p.allowed_emails)]

    if remaining_seconds > 0:
        # Poll still running -- send the user back to the dashboard with a notice.
        error = f"Time remaining: {remaining_str}"
        return render_template('dashboard.html', error=error, user=user,
                               allowed_polls=allowed_polls, polls=all_polls)

    # Poll has ended -- show the results.
    if not values or all(v == 0 for v in values):
        return render_template('poll_result.html', pie_chart_img=None, poll=poll,
                               user_id=user_id_field, total_votes=0,
                               remaining_time=remaining_str, leading_option=None,
                               leading_candidate_votes=0)

    leading_option = max(poll.options, key=poll.options.get)
    leading_candidate_votes = poll.options.get(leading_option)
    pie_chart_img = render_pie_chart(options, values)

    return render_template('poll_result.html', pie_chart_img=pie_chart_img, poll=poll,
                           user_id=user_id_field, total_votes=total_votes,
                           remaining_time=remaining_str, leading_option=leading_option,
                           leading_candidate_votes=leading_candidate_votes)


if __name__ == '__main__':
    app.run(debug=False)
