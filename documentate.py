import os
import re

os.system("rm -rf adaboost_bindings/target/doc")
os.system("rm -rf docs")
os.system("cd adaboost_bindings && cargo doc --no-deps && cd ..")

html_base_path = "adaboost_bindings/target/doc/adaboost_bindings/"
html_sub_paths = ["adaboost/struct.AdaBoost.html", "sample/struct.Sample.html",
                  "weak_learner/struct.WeakLearner.html", "weighted_data/struct.WeightedData.html"]

for html_sub_path in html_sub_paths:
    html_path = html_base_path + html_sub_path
    with open(html_path, "r") as f:
        content = f.read()

    match = re.search(r'<div id="trait-implementations-list">.*</div>', content, re.DOTALL)
    if match:
        trait_implementations_list = match.group(0)
        content = content.replace(trait_implementations_list, "")
    else:
        print("No match found.")

    match = re.search(r'<h2 id="trait-implementations".*/h2>', content, re.DOTALL)
    if match:
        trait_implementations = match.group(0)
        content = content.replace(trait_implementations, "")
    else:
        print("No match found.")

    match = re.search(r'<h3><a href="#trait-implementations">.*</section><h2><a href="index.html">', content, re.DOTALL)
    if match:
        blanket_implementations = match.group(0)
        content = content.replace(blanket_implementations, """</section><h2><a href="index.html">""")
    else:
        print("No match found.")

    with open(html_path, "w") as f:
        f.write(content)

os.system("mv adaboost_bindings/target/doc/ docs/")
os.system("""echo '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=adaboost_bindings/index.html"></head><body></body></html>' > docs/index.html""")