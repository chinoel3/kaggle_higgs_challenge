# Anacondaをインストール
- Download: https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation:how-do-i-verify-my-installers-integrity

# Anacondaで環境構築
**ターミナルを開けてanacondaを起動**
```
(base) [cshion@MacBook-Air-6 kaggle_higgs-boson]$
```
みたいに (base) がプロンプトの前についてたら成功。


**“ML”という名前の環境を作る**
```
conda create -n ML tensorflow pandas numpy matplotlib python=3.12.12
```
* `tensorflow`, `pandas`, `numpy` `matplotlib`をinstall
* `python`は書かなくても入るがversionを指定するために追加

**環境“ML”に入る**
```
conda activate ML
```

**追加で必要なパッケージをinstallする**
```
conda install -c conda-forge scikit-learn pydot graphviz
```
 * `conda-forge`という別のrepoを参照する必要がある

**環境から抜けたくなったら**
```
conda deactivate 
```

**今ある環境の一覧を表示**
```
conda info -e
```
目的に応じて環境を使い分けた方がよい（じゃないと一回インストールしたらバージョン管理が不可能になる）


# VS codeで走らせる
- Vidual Studio Code (https://code.visualstudio.com)
- Conda pathを通す: 左下の歯車 → `Settings` → 上の検索窓から`Conda path`と検索 → 入力
  * ターミナルから`which conda`と打って出てくるのがconda path
- ipynbファイルを開く
- 右上の`Select kernel` → `ML`を選ぶ。これでさっき作ったconda環境でスクリプトを走らせることができる。
  



