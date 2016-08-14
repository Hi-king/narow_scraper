# narow_scraper

生成
-----------

```bash
python generate.py --mid_size=300 --num_lstm_layer=4 models/139908304222248_25001.model.npz models/139908304222248_25001.converter.dump
# modelを作る時と同じパラメタを指定
```

学習
------------

webからのデータセット生成

```shell
python scraping.py
```

学習

```shell
python train.py --gpu=-1 --num_lstm_layer=4 --mid_size=300 --no_normalization
```