setup:
\tpip install -e . && pre-commit install

data:
\tpython -m ml_starter.scripts.download_data  # (เพิ่มทีหลังถ้าจะเขียนดาวน์โหลด zip)
train:
\tpython -m ml_starter.models.train
predict:
\tpython -m ml_starter.models.predict artifacts/model.joblib samples/one_row.json
test:
\tpytest -q
