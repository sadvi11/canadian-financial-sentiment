"""
Data Preparation — Canadian Financial Sentiment Classifier
Creates labeled training data, formats for BlazingText, uploads to S3.

BlazingText format: __label__positive TD Bank reports record profit
Labels: positive, negative, neutral
"""

import boto3
import os
import json
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")

# ── Training Data — Canadian Financial Headlines ───────────────────────────────

POSITIVE = [
    "TD Bank reports record Q1 2024 net income of 3.8 billion dollars",
    "Shopify shares surge 15 percent after beating revenue expectations",
    "RBC posts strong quarterly earnings driven by capital markets growth",
    "Bank of Canada signals potential rate cuts boosting housing market",
    "Canadian unemployment falls to 5.8 percent lowest in two years",
    "Suncor Energy increases quarterly dividend by 10 percent",
    "CN Rail reports strong earnings growth driven by bulk commodity volumes",
    "Canadian GDP grows 2.1 percent in Q4 beating analyst expectations",
    "BCE announces major 5G network expansion across Canadian cities",
    "Shopify merchant solutions revenue jumps 25 percent year over year",
    "TD Bank wealth management division reports record client assets",
    "Canadian housing starts rise sharply in March driven by Calgary and Toronto",
    "RBC acquires HSBC Canada expanding retail banking presence nationwide",
    "Bank of Nova Scotia reports higher than expected profits in Q2",
    "Canadian retail sales jump 1.4 percent in February beating forecasts",
    "Enbridge pipeline revenues exceed expectations on strong energy demand",
    "Canadian dollar strengthens against US dollar on oil price gains",
    "Manulife Financial reports strong insurance sales growth in Asia",
    "TMX Group reports record trading volumes on Canadian stock exchange",
    "Canadian oil sands production hits record levels boosting Alberta economy",
    "TD Bank digital banking customers surpass 15 million milestone",
    "Shopify announces profitable quarter for third consecutive time",
    "Sun Life Financial raises dividend citing strong earnings momentum",
    "Canadian manufacturing sector expands for fourth consecutive month",
    "RBC capital markets division posts record revenue in Q1",
    "Bank of Canada holds rates steady supporting economic stability",
    "Canadian home sales rise 8 percent in March from February",
    "Loblaw Companies beats earnings expectations on strong pharmacy sales",
    "Alimentation Couche-Tard reports record annual revenue globally",
    "Canadian fintech sector attracts record venture capital investment",
    "CIBC mortgage portfolio grows as housing demand recovers",
    "Telus announces strong wireless subscriber growth in Q1",
    "Canadian pension funds deliver above benchmark returns in 2023",
    "BMO Financial Group reports strong US banking integration results",
    "Canadian mining sector benefits from rising copper and gold prices",
    "Fortis utility company raises dividend for 50th consecutive year",
    "Canadian exports surge on strong global demand for resources",
    "Rogers Communications reports improved network performance metrics",
    "National Bank of Canada beats earnings estimates with strong retail growth",
    "Canadian venture capital investments reach record 7 billion in 2023",
    "TD Bank approved for US merger creating stronger North American presence",
    "Shopify capital program helps thousands of Canadian small businesses grow",
    "Canadian dollar rises to highest level in six months on oil rally",
    "CN Rail operating ratio improves to record low on efficiency gains",
    "Suncor achieves record oil sands production at lowest cost per barrel",
]

NEGATIVE = [
    "Canadian housing prices fall 8 percent year over year in major cities",
    "TD Bank faces US regulatory scrutiny over anti-money laundering failures",
    "Shopify cuts 20 percent of global workforce amid economic slowdown",
    "Canadian recession fears grow as GDP contracts for second quarter",
    "Bank of Canada raises rates to 5.25 percent causing mortgage payment shock",
    "RBC investment banking revenue falls sharply on deal slowdown",
    "Canadian inflation rises unexpectedly to 3.8 percent in October",
    "Suncor reports disappointing quarterly earnings on lower oil prices",
    "Canadian tech layoffs accelerate as US recession fears spread north",
    "BCE cuts dividend for first time in decades citing debt concerns",
    "Canadian commercial real estate vacancies hit record high in Toronto",
    "TD Bank pays 3 billion dollar fine for US money laundering violations",
    "Canadian consumer debt reaches record high as interest rates bite",
    "RBC warns of rising loan loss provisions as economy weakens",
    "Shopify stock falls 30 percent on weaker than expected guidance",
    "Canadian auto sector faces mass layoffs as EV transition accelerates",
    "Bank of Canada warns of elevated household financial stress levels",
    "Canadian housing starts fall sharply as developers pause new projects",
    "CIBC reports higher credit card delinquencies among Canadian consumers",
    "Canadian unemployment rises to 6.4 percent highest since 2021",
    "Enbridge pipeline project faces major regulatory rejection setback",
    "Canadian dollar weakens to lowest level in 18 months on oil slump",
    "Manulife reports investment losses on commercial real estate portfolio",
    "Canadian retail sales decline for third consecutive month",
    "Rogers outage leaves millions of Canadians without service for 19 hours",
    "Telus announces 6000 job cuts citing challenging economic environment",
    "Canadian mortgage arrears rise sharply as renewal shock hits borrowers",
    "BMO loan losses exceed analyst expectations on US commercial exposure",
    "Canadian energy sector faces headwinds from weak global demand",
    "Loblaw faces backlash and boycott over grocery price gouging concerns",
    "Bank of Nova Scotia cuts dividend growth plans amid economic uncertainty",
    "Canadian small businesses face record insolvencies in Q1 2024",
    "CN Rail reports volume decline on weak North American goods demand",
    "Canadian pension fund returns fall sharply on bond market losses",
    "Nortel style collapse feared as major Canadian tech firm files creditor protection",
    "Housing affordability crisis worsens as prices remain elevated despite rate hikes",
    "Canadian banks increase provisions for credit losses citing deteriorating outlook",
    "Suncor faces environmental liability charges in Alberta oilsands dispute",
    "Canadian dollar falls sharply on Bank of Canada dovish policy shift",
    "Major Canadian retailer files for bankruptcy protection amid competition",
    "TD Bank US expansion plans halted following regulatory order",
    "Canadian construction sector contracts as rate hikes freeze new builds",
    "Air Canada reports quarterly loss on fuel costs and labor disputes",
    "Canadian grain exports fall on weak global prices and logistics issues",
    "CIBC warns of significant credit quality deterioration in home equity loans",
]

NEUTRAL = [
    "Bank of Canada holds overnight rate at 5 percent for third meeting",
    "TD Bank announces Q2 2024 earnings release date of May 23",
    "Statistics Canada releases monthly employment report for April",
    "RBC opens new branch in downtown Calgary serving 500 clients daily",
    "Canadian government releases federal budget with infrastructure spending plans",
    "Shopify announces new merchant tools at annual Unite conference",
    "Bank of Canada governor speaks at Toronto business community luncheon",
    "TD Bank appoints new chief risk officer from internal candidates",
    "Statistics Canada reports housing price index for February",
    "RBC publishes annual corporate sustainability report",
    "Canadian Securities Administrators releases guidance on crypto assets",
    "CIBC holds annual general meeting in Toronto attended by shareholders",
    "Bank of Canada releases financial system review twice per year",
    "CN Rail announces annual shareholder meeting date and agenda",
    "Suncor publishes quarterly oil sands operational update",
    "Canadian banking regulator OSFI updates capital requirements framework",
    "TD Bank launches new mobile banking app update version 5.0",
    "Shopify reports monthly gross merchandise volume figures",
    "Bank of Nova Scotia announces executive leadership transition plan",
    "Canadian housing affordability report published by RBC economists",
    "BMO announces office relocation to new downtown Toronto headquarters",
    "Manulife holds investor day presenting five year strategic plan",
    "Canadian Mortgage and Housing Corporation releases rental market report",
    "Telus announces network infrastructure investment plan for 2024",
    "BCE files annual report with securities regulators on schedule",
    "Canadian trade data released by Statistics Canada showing monthly figures",
    "RBC economists publish quarterly Canadian economic outlook report",
    "TD Bank customer service center opens in Mississauga creating 500 jobs",
    "Fortis presents at annual utility investor conference in New York",
    "Canadian Bankers Association releases annual industry statistics report",
    "Enbridge holds conference call with analysts following quarterly results",
    "Bank of Canada releases monetary policy report quarterly update",
    "CIBC wealth management division publishes annual market outlook",
    "Canadian pension plan contribution rates set for 2025 announced",
    "Rogers announces network upgrade schedule for western Canadian provinces",
    "National Bank publishes research note on Canadian housing market trends",
    "Sun Life Financial holds annual meeting in Toronto on May 15",
    "Loblaw publishes corporate responsibility report for fiscal 2023",
    "Canadian dollar trading in narrow range as markets await Fed decision",
    "TMX Group releases monthly trading statistics for Canadian markets",
    "Alimentation Couche-Tard reports European acquisition integration progress",
    "Bank of Canada quarterly survey of business outlook released Wednesday",
    "Canadian government announces review of foreign investment guidelines",
    "Shopify hosts annual partner summit for Canadian developers",
    "Statistics Canada publishes retail trade survey results for January",
]


def create_blazingtext_data():
    """Format data for BlazingText supervised text classification."""
    all_data = []

    for text in POSITIVE:
        all_data.append(f"__label__positive {text.lower()}")
    for text in NEGATIVE:
        all_data.append(f"__label__negative {text.lower()}")
    for text in NEUTRAL:
        all_data.append(f"__label__neutral {text.lower()}")

    random.seed(42)
    random.shuffle(all_data)

    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    val_data = all_data[split:]

    print(f"Total examples: {len(all_data)}")
    print(f"Training: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Labels: positive={len(POSITIVE)}, negative={len(NEGATIVE)}, neutral={len(NEUTRAL)}")

    return train_data, val_data


def upload_to_s3(train_data, val_data):
    """Upload training data to S3."""
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Write temp files
    Path("train.txt").write_text("\n".join(train_data))
    Path("validation.txt").write_text("\n".join(val_data))

    prefix = "canadian-financial-sentiment/data"
    s3.upload_file("train.txt", S3_BUCKET, f"{prefix}/train.txt")
    s3.upload_file("validation.txt", S3_BUCKET, f"{prefix}/validation.txt")

    print(f"Uploaded to s3://{S3_BUCKET}/{prefix}/")
    print(f"Training URI: s3://{S3_BUCKET}/{prefix}/train.txt")
    print(f"Validation URI: s3://{S3_BUCKET}/{prefix}/validation.txt")

    # Cleanup
    Path("train.txt").unlink()
    Path("validation.txt").unlink()

    return f"s3://{S3_BUCKET}/{prefix}"


if __name__ == "__main__":
    print("Preparing Canadian Financial Sentiment training data...")
    train_data, val_data = create_blazingtext_data()

    print(f"\nSample training examples:")
    for ex in train_data[:3]:
        print(f"  {ex}")

    print(f"\nUploading to S3: {S3_BUCKET}...")
    data_uri = upload_to_s3(train_data, val_data)
    print(f"\nData ready at: {data_uri}")
    print("Run train.py next")
