from .db import db
import datetime

class Image(db.Document):
    location = db.StringField()
    name = db.StringField()


# pylint: disable=no-member
class Job(db.DynamicDocument):
    status = db.StringField(required=True)
    files = db.ListField(db.StringField())
    date_modified = db.DateTimeField(default=datetime.datetime.utcnow)


class Configuration(db.Document):
    name = db.StringField(primary_key=True)
    num_of_threads = db.IntField()
    headers = db.ListField(db.StringField())
    details = db.ListField(db.StringField())


def initialize_default_config():
    config = Configuration(name='basic',
                           num_of_threads=2,
                           headers=['S. No.', 'S. N.', 'S. No.', 'Sr. No.', 'Product ID', 'Item Code ', 'Cust Mat Code', 'Material Code',  # List of header attribute
                                    # make nw additions
                                    'Material Description', 'Part No', 'ISBN-13', 'Product Code', 'SKU', 'SKU\tSales Order/No/Pos/SeqHSN',
                                    'HSN Code', 'HSN/SAC', 'HSN of Goods/ Services', 'Title', 'Item Description', 'Description of Goods',
                                    'Desc of Goods/Services', 'Title & Author', 'Quantity', 'Quantity', 'QTY\tNo of packages',
                                    'SUPPL Qty', 'Unit Price', 'Mrp per Unit', 'Basic Rate', 'Unit Price\t',
                                    'Excise Duty', 'Freight'
                                    'Discount Percentage', 'Disc.', 'Disc. (INR)', 'Cash Discount (CD / amount * 100)SGST Percentage',
                                    'SGST/ UTGST', 'Tax %', 'Tax Rate', 'CGST Percentage', 'CGST Tax %', 'IGST Percentage', 'IGST Tax %',
                                    'Cess Percentage', 'Discount3', 'Discount2', 'Discount1'],
                           details=['Seller State', 'Separate Country from State', 'State', 'State Name', 'Place of Supply,',
                                    'State Name & CodeSeller ID', 'CIN No', 'PAN No', 'Seller Name', 'Header Parsing\t',
                                    'Registered Office Name & Address', 'Regd Office', 'Vendor Name', 'Billed By', 'Seller Address',
                                    'Vendor Address', 'Seller GSTN Number', 'GSTN No\t,GSTIN/UIN', 'GSTIN  No.', 'GST No', 'CIN',
                                    'Vendor GSTIN', 'GST Inv No', 'Our GSTIN', 'Country of Orgin', 'Along with State:', 'Currency',
                                    'Invoicing Currency', 'Description', 'Sale Order No', "Supplier's Ref", 'Payment Terms', 'Invoice Number',
                                    'Invoice No', 'Invoice No/Date', 'Invoice No/Series', 'Invoice Date', 'Invoice Date', 'Date', 'Due Date',
                                    'Due', 'Due Date', 'DatePayment Due Date', 'Payable On', 'Payable On or before', 'PO Number', 'Purchase Order No',
                                    'Customer PO No', 'PO Details', 'Supplier Ref', 'Invoice Items Total Amount', 'Buyer GSTN Number',
                                    'Contextual Analysis', 'GSTN No\t', 'GSTIN', 'GST Reg No', 'GSTIN/Unique ID', 'GSTIN ID',
                                    'Ship to Address\t', 'Ship To\tPlace of Supply\tDelivery Address', 'ship to', 'Sold to ', 'Name & Address',
                                    'Name of Customer (Billed to)', 'Buyer', 'Detail of Receiver', 'Consignee Name', 'Bill to Address', 'Ship To',
                                    'Name of Customer', 'Customer Name', 'delivery terms', 'invoice time', 'freight', 'delivery terms', 'state',
                                    'delivery date', 'bill'])
    config.id = 'basic'
    config.save()
