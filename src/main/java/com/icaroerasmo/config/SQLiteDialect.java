package com.icaroerasmo.config;

import org.hibernate.dialect.Dialect;

/**
 * Minimal SQLite dialect for Hibernate 6.x.
 * This is a very basic implementation suitable for simple use cases.
 */
public class SQLiteDialect extends Dialect {

    public SQLiteDialect() {
        super();
    }
}
