import re
from database.models import  Configuration




class Bbox(object):

    def __init__(self):
        config = Configuration.objects().get(name='basic')
        self.header = config.headers
        self.details_list = config.details
        self.weights_header = ['S. No.', 'S. N.', 'S. No.', 'Sr. No.','Description','duty','rate', 'HSN']
        self.header_ = ['S. No.', 'S. N.', 'S. No.', 'Sr. No.', 'Product ID', 'Item Code ', 'Cust Mat Code', 'Material Code',    ###List of header attribute
                        'Material Description', 'Part No', 'ISBN-13', 'Product Code', 'SKU', 'SKU\tSales Order/No/Pos/SeqHSN',  ## make nw additions
                        'HSN Code', 'HSN/SAC', 'HSN of Goods/ Services', 'Title', 'Item Description', 'Description of Goods',
                        'Desc of Goods/Services', 'Title & Author', 'Quantity', 'Quantity', 'QTY\tNo of packages',
                        'SUPPL Qty', 'Unit Price', 'Mrp per Unit', 'Basic Rate', 'Unit Price\t',
                         'Excise Duty', 'Freight'
                        'Discount Percentage', 'Disc.', 'Disc. (INR)', 'Cash Discount (CD / amount * 100)SGST Percentage',
                        'SGST/ UTGST', 'Tax %', 'Tax Rate', 'CGST Percentage', 'CGST Tax %', 'IGST Percentage', 'IGST Tax %',
                        'Cess Percentage', 'Discount3', 'Discount2', 'Discount1']
                        #  'TCS Percentage', 'Total tax/Total amount * 100', 'Grand Total - Basic Total', 'Total Amount',
                        # 'Net Payable','Ord.' 'Total (R/O)', 'Grand Total\t', 'APP%', 'Line Total', 'Total quantity pcs'
                        # ]
        self.details_list_ = ['Seller State', 'Separate Country from State', 'State', 'State Name', 'Place of Supply,', 
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
                        'Name of Customer', 'Customer Name', 'delivery terms', 'invoice time', 'freight','delivery terms', 'state',
                        'delivery date', 'bill', 'Customer']

        self.search_details = {'no' : ['Invoice Number', 'Invoice no', 'Invoice No'],
                               'date' : ['Invoice Date', 'Invoice Dt', 'Dated', 'Date'],
                                'Due' : ['Due', 'Due Date', 'Mode/Terms Of\n\nPayment',
                                         'Payment Due\n\nDate', 'payment terms'],
                                'gst' : ['GSTN', 'GSTN No.', 'GSTIN ID',
                                         'GSTIN/Unique ID', 'GSTN', 'GSTIN/UIN'],
                                'po' : ['PO No.'],
                                'ship' : ['Delivery Address','Bill to Address','Ship to', 'Customer name','Buyer','Ship', 'Buyers address',
                                          'Buyer\'s asdress', 'Address', 'Consignee', 'Billed Bygits ']
                            }

        self.remove_list =['id', 'basic', 'total', 'tax', '']
        # self.split_header()
        # print(self.header)
        # sys.exit()
        self.bbox = None
        self.header = [self.clean(head) for head in self.header]
        self.row_string = None
        self.similarity_index = None
        self.header_idx = None
        self.sheet = None
        self.info = None
        self.details = dict()
        self.details_status = []
        self.currency = 'INR'
        self.rows = None
        self.cols = None
        self.outside = None

    def clean(self, word):
        word = re.sub('[&()-/\.!@#$%*0-9\\t]', '', word)
        # word.replace('\n', ' ')
        return word.lower()
    def clean_details(self, word):
        word = re.sub('[&()@#$%*\\t\\n\']', '', word)
        return word.lower()

    def split_header(self):   ### optimization needed.

        """ Add optimization to header to increase accuracy 
        """
        temp = []
        for head in self.header:
            for _ in re.split('\\\\|/|\\ |\\\t|\\(|\\)', head):
                temp.append(_)

        temp = sorted(list(set(temp)))
        temp = [t for t in temp if t.lower() not in self.remove_list and len(t) > 1]
        self.header = temp