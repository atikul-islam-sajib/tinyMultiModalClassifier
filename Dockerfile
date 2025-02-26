FROM python:3.9-silim

WORKDIR /app

COPY . .

RUN pip install --no--cache--dir -r requirements.txt

ENTRYPOINT ["python", "src/cli.py"]