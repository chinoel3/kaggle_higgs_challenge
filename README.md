# Introduction (Higgs/ATLAS検出器)
https://drive.google.com/file/d/1G3f8xS6PMnNGI_XTix6FTS9qSVav5oZt/view?usp=sharing

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

# Training/Testデータのダウンロード
[ここ](https://drive.google.com/drive/folders/165-N-O6XOZJMZavvvPmChyzWF3U9H-Oz?usp=sharing)に置いてあります。
ダウンロードしてもらってもよいし、[テストスクリプト](https://github.com/HiggsIsBoson/kaggle_higgs_challenge/blob/main/test_DNN.ipynb)のようにコードの中にダウンロード操作を埋め込んでもらってもよいです。
```
import os, urllib.request
def download(url, output) : 
    if not os.path.exists(output):
        print(f"{output} not found. Downloading...")
        urllib.request.urlretrieve(url, output)
    else:
        print(f"{output} already exists. Skip download.")

download("https://drive.google.com/uc?id=1JPOVfYXJNXeBgG_V0auiuK5W2CvP4_td", "training.csv")
download("https://drive.google.com/uc?id=11W8QiL98fEqV7Xfbw-xgXCC7G-SM1yMb", "test.csv")
```



# VS codeで走らせる
- Vidual Studio Code (https://code.visualstudio.com)
- Conda pathを通す: 左下の歯車 → `Settings` → 上の検索窓から`Conda path`と検索 → 入力
  * ターミナルから`which conda`と打って出てくるのがconda path
- ipynbファイルを開く
- 右上の`Select kernel` → `ML`を選ぶ。これでさっき作ったconda環境でスクリプトを走らせることができる。

# データの構造
### Metadata
- `Label` : `s`がsignal, `b`がbackground.
- `Weight` : イベントにかけるべきweight (cross-sectionみたいなもの)
- `EventId` : イベントの番号  

### Event topology 
<img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/151793e2-f8d7-4548-b7c6-ec96d19c0299" />  

データに入っている $h \rightarrow \tau \tau$ 候補のイベント 
- いわゆる "lephad" チャンネル : leptonically decaying tau ($e/\mu$+neutrinos) + hadronically decaying tau ($\tau_h$)
- ATLAS jargonで, 再構成された"tau"というときは基本 $\tau_h$ のjetを指します。
- LeptonもLHCでは基本再構成された $e/\mu$ のことを指します。
- BGを落とすためにこの解析ではさらにinitial state radiation jet (ISR) という, high pTのjetがものがついているイベントだけを見ています。これがサンプルで"jet"と呼ばれてるやつです。複数いる場合もあって, dPhi_jet1_jet2みたいな変数はpTの1番目と2番目に高いjetの $\Delta \phi$ です。

Background イベント (BG) 
- 基本的に信号と同じ終状態のイベントがBGになります : $e/\mu$ が1つ + $\tau_h$ が1つ + jetが最低1つ + そこそこ立派なmissing ET
- 主な物理プロセス : $Z \rightarrow \tau\tau$, $t\bar{t} \rightarrow b\tau\nu b\tau\nu$, $W(\rightarrow e/\mu + \nu)$+fake $\tau_h$ (jetが誤って $\tau_h$ と同定されるもの)


### Kinematic variables 
- PRI_xxx :  low level feature
   * jet, tau, leptonとかの4-vector
   * `PRI_met` : Missing ET ("MET") の大きさ。pp衝突では陽子の内部要素 (gluonやquark : "parton") 同士の衝突になるので, 重心系の運動量はz方向には全く釣り合わず, 毎イベント前後方に大きくboostする。一方で陽子やpartonは横方向の運動量はほぼ0なので, 重心系は横方向には動いていない。よって終状態に出てきた粒子の横方向の運動量のsumは0となる。ニュートリノなどの検出されない粒子たちがイベントにいた場合は, 検出される粒子の運動量和の横方向成分にはインバランスが生じる。このインバランスがMETである。
   * `PRI_met_phi` : METの角度。
- DER_xxx : high level feature
   * `DER_mass_MMC` : $\tau$はneutrinoを出すので完全に再構成できないが、likelihoodを使って統計的に尤もらしいneutrinoの方向を決めて $\tau$ の4-vectorを再構成して組んだ,
      $\tau\tau$ のinvariant masss. $Z \rightarrow \tau\tau$ と $h \rightarrow \tau\tau$ を分離する上で最も強力。
   * `DER_mass_vis` : $\tau$ の崩壊で出てきたneutrino以外の粒子はvisible tauと呼ぶが, その2つのvisible tauのinvariant mass.
   * `DER_mass_transverse_met_lep` : METとleptonによるtransverse mass ($m_T$)。$m_T$ とは, 2つの粒子のpzをそれぞれ0と置いた時のinvariant mass。METのようなpz成分がわからないときにmassの情報を引き出すのに使う。Invariant mass $m_{inv}$ よりは必ず小さいので, $W \rightarrow \ell \nu$ みたいなイベントにおいて $m_T(\ell, \text{MET})$ みたいなものを計算するとW massでcut offを持つような分布になる。なので $W \rightarrow \ell \nu$ のようなBGを落とすのによく使われる。

### 分布をまずは書いて形を考えてみよう
- `draw_plots.ipynb` : `training.csv`の中のkinematic variableを全部書き出す。

# ML実装のminimal example
- `test_DNN.ipynb` : 3 layers densely connected DNN  

  <img width="300" height="300" alt="model_diagram" src="https://github.com/user-attachments/assets/1714906a-806b-4a98-9666-9f8f9bf06c38" />     
  
  * 分離能力はいかほどか？
  * Overtrainingはしているか？
  * Overtrainingしている気配がなかった場合はもっと複雑なモデルを使えるはずである。

# 感度の指標
- $\mathrm{AMS}=\sqrt{2\left(\left(s+b+b_r\right) \log \left(1+\frac{s}{b+b_r}\right)-s\right)}$
  * `s` : Scoreでカットした後のsignalのイベント数
  * `b` : Scoreでカットした後のBGのイベント数の期待値
  * `b_r` : `b`に対する系統効果。bの10%のときと, 30%のときで試してみましょう。
  * Weightをかける必要があることに注意
 
# レポートの提出
  * `test_DNN.ipynb`のようなコードを作る.
  * A4 1ページくらい簡単なレポートのPDFも提出する.
  * 自分のbranchを作ってこれらをpush
     * 自分のbranchを作成 : `git checkout -b submit-<family name>`
     * ファイルを追加 (`git add <files>`)
     * Commit : `git commit -m <comment>`
     * Repositoryにpush : `git push origin submit-<family name>`
  * 締切 : 2/27(金)

