# Tripletex Sandbox API Endpoints — Validated 2026-03-20

Sandbox: `https://kkpqfuj-amager.tripletex.dev/v2`

## Endpoint Status

| Endpoint | GET | POST | Notes |
|----------|-----|------|-------|
| `/employee` | 200 | 200 | Search by `firstName`, `lastName` |
| `/customer` | 200 | 200 | Search by `name` |
| `/product` | 200 | 200 | Search by `name` |
| `/invoice` | 422 without dates | 422 (needs bank account) | Requires `invoiceDateFrom`/`invoiceDateTo` params |
| `/project` | 200 | 200 | |
| `/department` | 200 | 200 | |
| `/contact` | 200 | 200 | |
| `/travelExpense` | 200 | 200 | |
| `/order` | 422 without dates | 200 | Requires `orderDateFrom`/`orderDateTo` |
| `/supplier` | 200 | - | |
| `/payment` | **404** | - | Does not exist as standalone endpoint |
| `/voucher` | **404** | - | Use `/ledger/voucher?dateFrom=X&dateTo=Y` instead |
| `/ledger/voucher` | 200 (with dates) | - | |

## Entity Field Structures (from sandbox)

### Employee
```
id, version, url, firstName, lastName, displayName, employeeNumber, dateOfBirth,
email, phoneNumberMobileCountry{id,url}, phoneNumberMobile, phoneNumberHome,
phoneNumberWork, nationalIdentityNumber, dnumber, internationalId{...},
bankAccountNumber, iban, bic, creditorBankCountryId, usesAbroadPayment,
userType, allowInformationRegistration, isContact, isProxy, comments,
address, department{id,url}, employments[{id,url}], holidayAllowanceEarned{...},
employeeCategory, isAuthProjectOverviewURL, pictureId, companyId, vismaConnect2FAactive
```

### Customer
```
id, version, url, name, organizationNumber, globalLocationNumber, supplierNumber,
customerNumber, isSupplier, isCustomer, isInactive, accountManager, department,
email, invoiceEmail, overdueNoticeEmail, phoneNumber, phoneNumberMobile,
description, language, displayName, isPrivateIndividual, singleCustomerInvoice,
invoiceSendMethod, emailAttachmentType, postalAddress{id,url},
physicalAddress{id,url}, deliveryAddress, category1-3, invoicesDueIn,
invoicesDueInType, currency{id,url}, bankAccountPresentation[],
ledgerAccount{id,url}, isFactoring, invoiceSendSMSNotification, ...
```

### Product
```
id, version, url, name, number, displayNumber, description, orderLineDescription,
costExcludingVatCurrency, costPrice, priceExcludingVatCurrency,
priceIncludingVatCurrency, isInactive, productUnit, incomingStock, outgoingStock,
vatType{id,url}, currency{id,url}, department, account, supplier, ...
```
**CONFIRMED: price field is `priceExcludingVatCurrency`**

### Department
```
id, version, url, name, departmentNumber, departmentManager, displayName,
isInactive, businessActivityTypeId
```

### Project
```
id, version, url, name, number, displayName, description, projectManager{id,url},
department, mainProject, startDate, endDate, customer, isClosed,
isReadyForInvoicing, isInternal, isOffer, isFixedPrice, projectCategory, ...
```

### Travel Expense
```
id, version, url, attestationSteps[], attestation, project, employee{id,url},
approvedBy, completedBy, rejectedBy, department{id,url}, payslip,
vatType{id,url}, paymentCurrency{id,url}, voucher, attachment,
isCompleted, isApproved, rejectedComment, isChargeable, completedDate,
approvedDate, date, travelAdvance, fixedInvoicedAmount, amount,
chargeableAmountCurrency, paymentAmount, title, ...
```
**CONFIRMED: `employee.firstName` works via field expansion `fields=employee(firstName)`**
**CONFIRMED: `title` field exists**

### Order
```
id, version, url, customer{id,url}, contact, attn, displayName, receiverEmail,
number, reference, department, orderDate, project, invoiceComment,
currency{id,url}, invoicesDueIn, invoicesDueInType, isClosed, deliveryDate,
deliveryAddress, deliveryComment, orderLines[], ...
```

## Key Findings

1. **`/payment` does not exist** — payment registration must go through invoice workflow
2. **Invoice/order listing requires date params** — `invoiceDateFrom`/`invoiceDateTo` and `orderDateFrom`/`orderDateTo` are mandatory
3. **Invoice creation requires bank account** — sandbox needs bank account setup first
4. **Nested objects expandable** — use `fields=employee(firstName,lastName)` to expand nested refs
5. **`/voucher` → `/ledger/voucher`** with date params required
