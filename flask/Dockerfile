FROM python:3.7

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3

COPY . .

# run program
CMD ["python", "./app.py"]
