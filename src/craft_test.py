import re

# li = ['S. No.', 'S. N.', 'S. No.', 'Sr. No.', 'Product ID', 'Item Code ', 'Cust Mat Code', 'Material Code',
#  'Material Description', 'Part No', 'ISBN-13', 'Product/Color Code', 'SKU', 'SKU\tSales Order/No/Pos/SeqHSN',
#   'HSN Code', 'HSN/SAC', 'HSN of Goods/ Services', 'Title', 'Item Description', 'Description of Goods',
#    'Desc of Goods/Services', 'Title & Author', 'Quantity', 'Quantity', 'Total quantity pcs', 'QTY\tNo of packages',
#     'SUPPL Qty', 'Unit Price', 'Mrp per Unit', 'Basic Rate', 'Unit Price\t', 'Excise Duty', 'Freight',
#      'Discount Percentage', 'Disc.', 'Disc. (INR)', 'Cash Discount (CD / amount * 100)SGST Percentage',
#       'SGST/ UTGST', 'Tax %', 'Tax Rate', 'CGST Percentage', 'CGST Tax %', 'IGST Percentage', 'IGST Tax %',
#        'Cess Percentage', 'TCS Percentage', 'Total tax/Total amount * 100', 'Grand Total - Basic Total', 'Total Amount',
#         'Net Payable', 'Total (R/O)', 'Grand Total\t', 'APP%', 'Line Total']


li = 	['Seller State', 'Separate Country from State', 'State', 'State Name', 'Place of Supply,', 
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
         'Name of Customer', 'Name']

# li = [re.sub('[&()-/\.!@#$%*0-9 \\t\\n]', '', l) for l in li]
# li = [l.lower() for l in li]

print(li)
