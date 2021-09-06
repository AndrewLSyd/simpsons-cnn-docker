# curl -fsSL https://get.docker.com -o get-docker.sh
# sudo sh get-docker.sh
FROM python:3.7.3

COPY . /

RUN pip install --no-cache-dir -r requirements-pip.txt

ENTRYPOINT ["python", "/score.py"]

CMD [ "test_new" ]

# RUN apt-get update && apt-get install -y && cd /app && bash 01_setup_with_requirement_files.sh
# CMD cd /app; bash score.sh
