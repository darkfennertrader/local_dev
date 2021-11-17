FROM 88178d65d12c

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN mkdir /home/dev_env

WORKDIR /home/dev_env

COPY . .

CMD ["tail","-f","/dev/null"]