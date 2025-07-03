from pydantic import BaseModel, Field
from enum import Enum


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
    is_invoice: float = Field(description="What is a probability of this beeing invoice. Number form 0 to 1.")
    
    order_total_price: float = Field(description="Total price of the order")
    order_currency: Currency = Field(description="Currency of the order")
    purchase_order_id: str = Field(description="Unique identification of the order for the customer. This could be a purchase order number or an external order number, e.g., DNO25052613. Purchase order ID starts with 'DNO' and is followed by the date (YYMMDD) and a sequence number (two digits). This ID may be hand written on the invoice in its top corner.")

    items: list[InvoceItem] = Field(description="List of items")
