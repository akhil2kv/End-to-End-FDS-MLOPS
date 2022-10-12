# Transaction data simulator

This section presents a transaction data simulator of legitimate and fraudulent transactions. This simulator will be used throughout the rest of this book to motivate and assess the efficiency of different fraud detection techniques in a reproducible way.

A simulation is necessarily an approximation of reality. Compared to the complexity of the dynamics underlying real-world payment card transaction data, the data simulator that we present below follows a simple design.

This simple design is a choice. First, having simple rules to generate transactions and fraudulent behaviors will help in interpreting the kind of patterns that different fraud detection techniques can identify. Second, while simple in its design, the data simulator will generate datasets that are challenging to deal with.

The simulated datasets will highlight most of the issues that practitioners of fraud detection face using real-world data. In particular, they will include class imbalance (less than 1% of fraudulent transactions), a mix of numerical and categorical features (with categorical features involving a very large number of values), non-trivial relationships between features, and time-dependent fraud scenarios.

## Design choices

### Transaction features
Our focus will be on the most essential features of a transaction. In essence, a payment card transaction consists of any amount paid to a merchant by a customer at a certain time. The six main features that summarise a transaction therefore are:

- The transaction ID: A unique identifier for the transaction
- The date and time: Date and time at which the transaction occurs
- The customer ID: The identifier for the customer. Each customer has a unique identifier
- The terminal ID: The identifier for the merchant (or more precisely the terminal). Each terminal has a unique identifier
- The transaction amount: The amount of the transaction.
- The fraud label: A binary variable, with the value 00 for a legitimate transaction, or the value 11 for a fraudulent transaction.
These features will be referred to as ` TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID, TERMINAL_ID, TX_AMOUNT, and TX_FRAUD`

