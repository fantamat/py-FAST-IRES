import os

from PIL import Image
from google import genai

from test_utility import test_all

from pydantic import BaseModel, Field
from enum import Enum

"""
ordered  = (ALIEN / DEYMED)


billing_account_name
billing_address_street
billing_address_city
billing_address_postalcode
billing_address_country
billing_account_company_id
billing_account_vat_id

items:
pn
name
quantity
unit_price
total_price

order_total_price
order_currency
"""

class Currency(str, Enum):
    CZK = "CZK"
    EUR = "EUR"
    USD = "USD"
    OTHER = "Other"


class Customer(str, Enum):
  DEYMED = "Deymed"
  ALIEN = "Alien"
  OTHER = "Other"
  NONE = "None"


class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City")
    postalcode: str = Field(description="Postal code")
    country: str = Field(description="Country")


class BillingAccount(BaseModel):
    account_name: str = Field(description="Supplier name. Unique identification of the company for the customer")
    company_id: str = Field(description="Supplier company ID. Unique identification of the company for the state, e.g., IČO in Czechia")
    vat_id: str = Field(description="Supplier VAT ID. Unique international identification of the company for the state, e.g., DIČ in Czechia")
    adress: Address = Field(description="Supplier address")
    account_phone: str = Field(description="Supplier phone")
    account_email: str = Field(description="Supplier email")


class InvoceItem(BaseModel):
    part_number: str = Field(description="Part number of the item, if applicable")
    description: str = Field(description="Description of the item")
    quantity: int = Field(description="Quantity of the item")
    unit_price: float = Field(description="Unit price of the item")
    total_price: float = Field(description="Total price of the item")


class Invoice(BaseModel):
    customer: Customer = Field(description="Customer name. Unique identification of the company for the supplier")
    billing_account: BillingAccount = Field(description="Billing account. Unique identification of the company for the customer")
    
    order_total_price: float = Field(description="Total price of the order")
    order_currency: Currency = Field(description="Currency of the order")
    
    items: list[InvoceItem] = Field(description="List of items")



def main():
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def process_image(image_path):

        image = Image.open(image_path)
        # response = client.models.generate_content(
        #     model="gemini-2.5-pro-exp-03-25",
        #     contents=[image, "Extract the structured data from the image in the JSON format."],
        # )

        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25",
            contents=[image, "Extract the structured data from the image in the given JSON format."],
            config={
                'response_mime_type': 'application/json',
                'response_schema': Invoice,
            },
        )
        invoice: Invoice = response.parsed
        return {
            "invoice": invoice.model_dump(),
            "total_token_count": response.usage_metadata.total_token_count,
        }

    test_all(process_image, "data/outputs/gemini-pro-preview/")


if __name__ == "__main__":
    main()