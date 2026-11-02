# 第2回・基礎の実行準備

**AI支援で準備した未実行の資料です。本人はまだ内容確認・学習実行をしていません。基礎課題の実験は未完了で、後で学習する予定です。**

## 準備した工夫

- 変更を `num_train_epochs: 20 → 10` の1項目に限定しました。モデル、データ、seed、学習率、LoRA rankは同じです。
- 学習時間と最終ログlossを結果に保存する処理を追加しました。公式の `before.seconds` / `after.seconds` は評価時間なので、学習時間には使いません。
- セッション再利用を検知して停止し、前のadapterの持ち越しを防ぎます。
- 学習全体の平均lossと、最後に記録されたlossを区別して保存します。

これらはCodexが準備・ローカル検証した内容で、本人の実験成果ではありません。ローカルの14テストとコードセルの構文確認は成功しました。GPU学習そのものは検証していません。

## 後で実行する手順

1. Colabの「ノートブックを開く → アップロード」で `notebooks/lec02_baseline20.ipynb` を開きます。
2. 無料のT4 GPUを選び、上から全セルを実行します。有料への変更はしません。Gemini APIキーは不要です。
3. 最後にダウンロードされる `baseline20-results.zip` を保存します。
4. セッションを再起動し、`notebooks/lec02_epochs10.ipynb` を新たに開いて同様に実行します。`epochs10-results.zip` を保存します。
5. 2つの `result.json` のformat_rate、content_rate、`training.final_logged_loss`、`training.train_seconds`を比較します。実際の出力と失敗例も読み、本人の考察を加えます。
6. 終了後はランタイムを接続解除します。

将来、本人が実行する場合だけ `EXECUTION_ACTOR="user"` を使います。エージェントが実行する場合は `"assistant"` に変更します。現在は未実行なので、本人の実測結果はありません。

エポック数を変えると、固定したwarmup_steps=6の総ステップに対する割合も15%から30%へ変わります。結果を学習量だけの効果とは断定しません。

## 出典

- [公式第2回課題](https://github.com/matsuolab/lecture-ai-engineering-2026/blob/fae899a409e2c99feee9d44a3789f936f5fc64ee/lectures/02/problem.md)
- [公式ノートブック](https://github.com/matsuolab/lecture-ai-engineering-2026/blob/fae899a409e2c99feee9d44a3789f936f5fc64ee/notebooks/lora_format_tuning.ipynb)

実行用ノートブックはこの公式版のセクション0〜7をもとに、上記の保存補助と安全な条件切替を追加しました。任意の4bit観察は含めていません。
