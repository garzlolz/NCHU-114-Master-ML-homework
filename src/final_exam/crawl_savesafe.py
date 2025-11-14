import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime

# 分類與對應 t_s_id
categories = {
    "米油罐頭泡麵": [
        ("米/五穀/濃湯", "42649"),
        ("油/調味料", "42652"),
        ("泡麵 / 麵條", "42660"),
        ("罐頭調理包", "42644"),
    ],
    "餅乾零食飲料": [
        ("休閒零嘴", "39140"),
        ("美味餅乾", "39198"),
        ("糖果/巧克力", "62362"),
        ("飲料", "39153"),
    ],
    "奶粉養生保健": [
        ("養生保健/常備品", "43047"),
        ("奶粉/穀麥片", "43046"),
        ("特色茶品", "43053"),
        ("咖啡/可可", "43049"),
    ],
    "沐浴開架保養": [
        ("沐浴乳香皂", "42661"),
        ("美髮造型", "42640"),
        ("口腔清潔", "42638"),
        ("臉部清潔", "42659"),
        ("開架/身體保養", "42643"),
    ],
    "餐廚衛浴居家": [
        ("鍋具/飲水/廚房", "39213"),
        ("掃除用具/照明/五金", "44678"),
        ("傢飾/收納/衛浴", "39202"),
        ("寵物/園藝", "39189"),
    ],
    "日用清潔用品": [
        ("衛生紙/濕巾", "42657"),
        ("衣物清潔", "42637"),
        ("居家清潔", "42642"),
        ("衛生棉/護墊", "54722"),
        ("成人/嬰兒紙尿褲", "54952"),
    ],
    "家電/3C配件": [
        ("廚房家電", "39220"),
        ("季節家電", "39197"),
        ("生活家電", "47024"),
        ("3C/電腦周邊/OA", "39172"),
    ],
    "文具休閒服飾": [
        ("文具/辦公用品", "57242"),
        ("汽機車百貨", "39233"),
        ("休閒/運動", "39203"),
        ("服飾/配件", "67217"),
    ],
    "免運/主題專區": [
        ("冷凍免運", "67217"),
        ("箱購免運", "67742"),
        ("主題專區", "67745"),
    ],
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
}


def fetch_product_detail(product_url, product_name, max_retries=3):
    """爬取商品詳細頁，加入重試機制和詳細 log"""
    for attempt in range(max_retries):
        try:
            print(
                f"  → Fetching detail for: {product_name[:30]}... (attempt {attempt + 1}/{max_retries})"
            )
            r = requests.get(product_url, headers=headers, timeout=10)
            if r.status_code != 200:
                print(f"    ✗ Failed with status code: {r.status_code}")
                return ""
            soup = BeautifulSoup(r.text, "html.parser")

            # 正確的選擇器：ProductDescriptionListArea
            detail_section = soup.select_one("div.ProductDescriptionListArea")

            if not detail_section:
                # 備選方案
                detail_section = soup.select_one("div.ProductDescriptionArea")

            if detail_section:
                # 取得所有文字內容
                detail_text = detail_section.get_text(separator="\n", strip=True)
                print(f"    ✓ Successfully fetched detail ({len(detail_text)} chars)")
                return detail_text
            else:
                print(f"    ⚠ No detail section found")
                return ""
        except requests.exceptions.Timeout:
            print(f"    ⏱ Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    ✗ Failed after {max_retries} attempts")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Error: {e}")
            return ""


def fetch_products(category_id, category_name, subcategory_name):
    base_url = "https://www.savesafe.com.tw/Products/ProductList"
    page = 1
    products = []

    while True:
        try:
            params = {"t_s_id": category_id, "Pg": page, "s": 6}
            r = requests.get(base_url, params=params, headers=headers, timeout=10)
            if r.status_code != 200:
                print(
                    f"Failed to fetch page {page} from {subcategory_name} (Category: {category_name}), status code: {r.status_code}"
                )
                break

            soup = BeautifulSoup(r.text, "html.parser")
            product_blocks = soup.select("div.col.mb-4.text-left.NewActivityItem")
            if not product_blocks:
                print(
                    f"No products found on page {page} for {subcategory_name} (Category: {category_name}), stopping."
                )
                break

            print(f"Processing page {page} with {len(product_blocks)} products...")

            for idx, block in enumerate(product_blocks, 1):
                sku = (
                    block.select_one("input#data_Prd_Sku")["value"]
                    if block.select_one("input#data_Prd_Sku")
                    else ""
                )
                attr_no = (
                    block.select_one("input#data_Prd_Attribute_Item_No")["value"]
                    if block.select_one("input#data_Prd_Attribute_Item_No")
                    else ""
                )
                prdatt_sid = (
                    block.select_one("input#PrdAtt_SID")["value"]
                    if block.select_one("input#PrdAtt_SID")
                    else ""
                )
                img_tag = block.select_one("img.card-img-top")
                img_url = img_tag["src"] if img_tag else ""
                link_tag = block.select_one('a[href^="ProductView"]')
                if link_tag:
                    href = link_tag["href"]
                    if not href.startswith("/"):
                        href = "/" + href
                    if href.startswith("/ProductView"):
                        href = "/Products" + href
                    link = "https://www.savesafe.com.tw" + href
                else:
                    link = ""
                name_tag = block.select_one("p.card-title.ItemName")
                name = name_tag.text.strip() if name_tag else ""
                description_tag = block.select_one("p.mb-2.ObjectName")
                description = description_tag.text.strip() if description_tag else ""
                price_tag = block.select_one("span.Price")
                price = price_tag.text.strip() if price_tag else ""

                print(f"[{idx}/{len(product_blocks)}] Processing: {name[:40]}...")
                description_detail = fetch_product_detail(link, name) if link else ""

                products.append(
                    {
                        "sku": sku,
                        "attribute_no": attr_no,
                        "prdatt_sid": prdatt_sid,
                        "name": name,
                        "description": description,
                        "description_detail": description_detail,
                        "price": price,
                        "image_url": img_url,
                        "product_link": link,
                        "category": category_name,
                        "subcategory": subcategory_name,
                    }
                )

            print(
                f"✓ Fetched {len(product_blocks)} products from page {page} of {subcategory_name} (Category: {category_name})"
            )

            next_page_link = soup.select_one(f'a[href*="Pg={page+1}"]')
            if not next_page_link:
                print(
                    f"No next page for {subcategory_name} (Category: {category_name}), done."
                )
                break

            page += 1
            time.sleep(1)

        except requests.exceptions.Timeout:
            print(f"Timeout fetching page {page}, retrying...")
            time.sleep(3)
            continue
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break

    return products


def save_to_csv_append(products, filename, write_header=False):
    """追加模式寫入 CSV"""
    if not products:
        print("No data to save.")
        return
    keys = products[0].keys()
    mode = "a" if not write_header else "w"
    with open(filename, mode, newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, keys)
        if write_header:
            dict_writer.writeheader()
        dict_writer.writerows(products)
    print(f"💾 Saved {len(products)} products to {filename}\n")


if __name__ == "__main__":
    # 生成帶時間戳記的檔案名稱
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./output/savesafe_products_{timestamp}.csv"

    print(f"\n{'='*80}")
    print(f"SaveSafe Product Crawler")
    print(f"Output file: {filename}")
    print(f"{'='*80}\n")

    first_write = True
    total_products = 0

    for category_name, subcats in categories.items():
        for subcat_name, t_s_id in subcats:
            print(f"\n{'='*80}")
            print(f"Start crawling {subcat_name} under {category_name}")
            print(f"{'='*80}")
            prods = fetch_products(t_s_id, category_name, subcat_name)

            # 每個子分類爬完就寫入
            save_to_csv_append(prods, filename, write_header=first_write)
            first_write = False
            total_products += len(prods)

    print(f"\n{'='*80}")
    print(f"✓ Crawling completed! Total products: {total_products}")
    print(f"✓ Data saved to: {filename}")
    print(f"{'='*80}")
