docker compose build --no-cache trainer-gpu

Run fetch_bdg2.py:

docker compose run --rm trainer-gpu bash -lc ^
docker compose run --rm trainer-gpu bash -lc "python -m src.ingest.fetch_bdg2 --out_root /app/data/external/bdg2 --emit_parquet /app/data/processed/bdg2_electricity_long.parquet"


Train model:

docker compose up trainer-gpu

Forecating model

docker compose run --rm trainer-gpu bash -lc "
python -m src.forecasting.forecast_catboost \
  --parquet /app/data/processed/bdg2_electricity_long.parquet \
  --model  /app/models/catboost_gpu.cbm \
  --horizon 168 --freq H \
  --lags 1,24,168 --roll_windows 24,168 \
  --out /app/data/forecasts/cb_forecast.parquet
"


Create dashboard

docker compose up dash



