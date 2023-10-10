# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from flask import render_template, redirect, request, url_for
from flask_login import (
    current_user,
    login_user,
    logout_user
)

from apps import db, login_manager
from apps.authentication import blueprint
from apps.authentication.forms import LoginForm, CreateAccountForm
from apps.authentication.models import Users

from apps.authentication.util import verify_pass





# Login & Registration

@blueprint.route('/login', methods=['GET', 'POST'])
def login():
    login_form = LoginForm(request.form)
    create_account_form = CreateAccountForm(request.form)

    if request.method == 'POST':
        if 'register' in request.form:
            # Handle registration form (form2)
            username = request.form['username']
            email = request.form['email']

            # Check username exists
            user_username = Users.query.filter_by(username=username).first()
            if user_username:
                return render_template('login.html',
                                       form1=create_account_form,
                                       form2=login_form,
                                       msg='Username already registered',
                                       success=False)

            # Check email exists
            user_email = Users.query.filter_by(email=email).first()
            if user_email:
                return render_template('login.html',
                                       form1=create_account_form,
                                       form2=login_form,
                                       msg='Email already registered',
                                       success=False)

            # Create the user
            user = Users(**request.form)
            db.session.add(user)
            db.session.commit()

            return render_template('login.html',
                                   form1=create_account_form,
                                   form2=login_form,
                                   msg='User created, please login',
                                   success=True)

        elif 'login' in request.form:
            # Handle login form (form1)
            username = request.form['username']
            password = request.form['password']

            # Locate user
            user = Users.query.filter_by(username=username).first()

            # Check the password
            if user and verify_pass(password, user.password):
                login_user(user)
                return redirect(url_for('home_blueprint.Dashboard'))
            else:
                # Something (user or pass) is not ok
                return render_template('login.html',
                                       form1=create_account_form,
                                       form2=login_form,
                                       msg='Wrong user or password',
                                       success=False)

    elif not current_user.is_authenticated:
        # Show login form by default
        return render_template('login.html',
                               form1=create_account_form,
                               form2=login_form)

    return redirect(url_for('home_blueprint.Dashboard'))


@blueprint.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('authentication_blueprint.login'))


# Errors

@login_manager.unauthorized_handler
def unauthorized_handler():
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('home/page-403.html'), 403


@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('home/page-404.html'), 404


@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('home/page-500.html'), 500
