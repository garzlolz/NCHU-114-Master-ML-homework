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
}

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/98.0.4758.102 Safari/537.36"
    )
}


def fetch_product_detail(product_url, product_name, max_retries=3):
    """
    抓取商品詳細頁的描述資訊

    Args:
        product_url: 商品詳細頁網址
        product_name: 商品名稱（用於日誌）
        max_retries: 最大重試次數

    Returns:
        str: 商品詳細描述 (description)
    """
    for attempt in range(max_retries):
        try:
            print(
                f"  取得商品詳細資訊：{product_name[:30]}... "
                f"(第 {attempt + 1}/{max_retries} 次嘗試)"
            )
            r = requests.get(product_url, headers=headers, timeout=10)
            if r.status_code != 200:
                print(f"    請求失敗，狀態碼：{r.status_code}")
                return ""

            soup = BeautifulSoup(r.text, "html.parser")

            # 商品說明區塊選擇器
            detail_section = soup.select_one("div.ProductDescriptionListArea")

            if not detail_section:
                # 備用商品說明區塊
                detail_section = soup.select_one("div.ProductDescriptionArea")

            if detail_section:
                # 取得所有文字內容作為詳細描述
                detail_text = detail_section.get_text(separator="\n", strip=True)
                print(f"    成功取得商品描述（{len(detail_text)} 字元）")
                return detail_text
            else:
                print("    找不到商品描述區塊")
                return ""

        except requests.exceptions.Timeout:
            print(f"    連線逾時，第 {attempt + 1}/{max_retries} 次")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    多次重試後仍逾時，放棄此商品")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"    請求錯誤：{e}")
            return ""


def fetch_products(category_id, category_name, subcategory_name):
    """
    依分類抓取商品列表

    欄位說明：
    - brand: 品牌名稱 (ItemName)
    - name: 商品名稱 (ObjectName)
    - description: 詳細描述 (ProductDescriptionListArea)
    """
    base_url = "https://www.savesafe.com.tw/Products/ProductList"
    page = 1
    products = []

    while True:
        try:
            params = {"t_s_id": category_id, "Pg": page, "s": 6}
            r = requests.get(base_url, params=params, headers=headers, timeout=10)
            if r.status_code != 200:
                print(
                    f"無法取得第 {page} 頁：{subcategory_name}"
                    f"（大分類：{category_name}），狀態碼：{r.status_code}"
                )
                break

            soup = BeautifulSoup(r.text, "html.parser")
            product_blocks = soup.select("div.col.mb-4.text-left.NewActivityItem")

            if not product_blocks:
                print(
                    f"{subcategory_name}（大分類：{category_name}）"
                    f"在第 {page} 頁沒有商品，停止爬取。"
                )
                break

            print(f"處理第 {page} 頁，共 {len(product_blocks)} 筆商品資料...")

            for idx, block in enumerate(product_blocks, 1):
                # 商品基本資訊
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

                # 圖片網址
                img_tag = block.select_one("img.card-img-top")
                img_url = img_tag["src"] if img_tag else ""

                # 商品連結
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

                # 品牌
                brand_tag = block.select_one("p.card-title.ItemName")
                brand = brand_tag.text.strip() if brand_tag else ""

                # 商品名稱
                name_tag = block.select_one("p.mb-2.ObjectName")
                name = name_tag.text.strip() if name_tag else ""

                price_tag = block.select_one("span.Price")
                price = price_tag.text.strip() if price_tag else ""

                print(f"[{idx}/{len(product_blocks)}] 處理商品：{name[:40]}...")
                description = fetch_product_detail(link, name) if link else ""

                products.append(
                    {
                        "sku": sku,
                        "attribute_no": attr_no,
                        "prdatt_sid": prdatt_sid,
                        "brand": brand,  # 品牌 (ItemName)
                        "name": name,  # 商品名稱 (ObjectName)
                        "description": description,  # 詳細描述 (ProductDescriptionListArea)
                        "price": price,
                        "image_url": img_url,
                        "product_link": link,
                        "category": category_name,
                        "subcategory": subcategory_name,
                    }
                )

            print(
                f"已從 {subcategory_name}（大分類：{category_name}）"
                f"第 {page} 頁抓取 {len(product_blocks)} 筆商品。"
            )

            # 檢查是否有下一頁
            next_page_link = soup.select_one(f'a[href*="Pg={page+1}"]')
            if not next_page_link:
                print(
                    f"{subcategory_name}（大分類：{category_name}）"
                    f"沒有下一頁，完成此子分類。"
                )
                break

            page += 1
            time.sleep(1)

        except requests.exceptions.Timeout:
            print(f"抓取第 {page} 頁時連線逾時，稍後重試...")
            time.sleep(3)
            continue
        except Exception as e:
            print(f"第 {page} 頁發生錯誤：{e}")
            break

    return products


def save_to_csv_append(products, filename, write_header=False):
    """以追加模式寫入 CSV 檔案"""
    if not products:
        print("沒有資料可寫入。")
        return

    keys = products[0].keys()
    mode = "a" if not write_header else "w"

    with open(filename, mode, newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, keys)
        if write_header:
            dict_writer.writeheader()
        dict_writer.writerows(products)

    print(f"已將 {len(products)} 筆商品寫入 {filename}\n")


if __name__ == "__main__":
    # 產生含時間戳記的輸出檔名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./output/savesafe_products_{timestamp}.csv"

    print("\n" + "=" * 80)
    print("SaveSafe 商品爬蟲")
    print(f"輸出檔案：{filename}")
    print("=" * 80)
    print("\n欄位說明：")
    print("  - brand: 品牌名稱 (HTML: ItemName)")
    print("  - name: 商品名稱 (HTML: ObjectName)")
    print("  - description: 詳細描述 (HTML: ProductDescriptionListArea)")
    print("=" * 80 + "\n")

    first_write = True
    total_products = 0

    for category_name, subcats in categories.items():
        for subcat_name, t_s_id in subcats:
            print("\n" + "=" * 80)
            print(f"開始爬取：{category_name} - {subcat_name}")
            print("=" * 80)

            prods = fetch_products(t_s_id, category_name, subcat_name)

            # 每個子分類爬完就寫入檔案
            save_to_csv_append(prods, filename, write_header=first_write)
            first_write = False
            total_products += len(prods)

    print("\n" + "=" * 80)
    print(f"全部爬取完成！總商品數：{total_products}")
    print(f"資料已寫入：{filename}")
    print("=" * 80)
