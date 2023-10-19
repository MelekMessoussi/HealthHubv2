
## ✨ How to use it

> Download the code 

```bash
$ git clone https://github.com/yahyasamet/HealthHubv2.git
$ cd HealthHubv2
```

<br />

### 👉 Set Up for `Windows` 

> Install modules via `VENV` (windows) 

```
$ python venv venv
$ .\env\Scripts\activate
$ pip install -r requirements.txt
$ git clone https://github.com/NVlabs/stylegan3

and you should install the Gan model from here https://huggingface.co/spaces/yahyasmt/Brain-MR-Image-Generation-with-StyleGAN/tree/main
you copy the file brainmrigan.pkl and make it in to HealthHubv2 folder
```

<br />

> Set Up Flask Environment

```bash
$ # CMD 
$ set FLASK_APP=run.py
$ set FLASK_ENV=development
$
$ # Powershell
$ $env:FLASK_APP = ".\run.py"
$ $env:FLASK_ENV = "development"
```

<br />

> Start the app

```bash
$ flask run
```

At this point, the app runs at `http://127.0.0.1:5000/`. 

<br />
