from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)
df = load_data()
df["length"] = df.message.str.len()
df["words"] = df.message.str.split().str.len()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="label")
plt.title("Ham vs Spam Messages")
plt.tight_layout(); plt.savefig(OUT / "class_distribution.png", dpi=160); plt.close()

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="length", hue="label", bins=50, element="step", stat="density", common_norm=False)
plt.title("Message Length Distribution")
plt.tight_layout(); plt.savefig(OUT / "message_length.png", dpi=160); plt.close()

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="label", y="words")
plt.title("Word Count by Class")
plt.tight_layout(); plt.savefig(OUT / "word_count.png", dpi=160); plt.close()
print(f"EDA charts saved in {OUT}")
