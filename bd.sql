drop table Patient;
drop table Hospital;

CREATE TABLE Patient(
    id_number CHAR(14) NOT NULL PRIMARY KEY,
    name VARCHAR2(255) NOT NULL,
    cpf CHAR(11) NOT NULL UNIQUE,
    rg CHAR(9) NOT NULL UNIQUE,
    date_of_birth DATE NOT NULL,
    agree VARCHAR(26) NOT NULL,
    local VARCHAR(4),
    dependent_number CHAR(14),

    CONSTRAINT FK_dependent_number FOREIGN KEY (dependent_number) REFERENCES Patient(id_number) ON DELETE CASCADE
);

CREATE TABLE Hospital(
    id_hospital CHAR(14),
    local VARCHAR(4),
    psychiatrist NUMBER(1) DEFAULT 0 NOT NULL,
    dermatologist NUMBER(1) DEFAULT 0 NOT NULL,
    cardiologist NUMBER(1) DEFAULT 0 NOT NULL,
    general_practitioner NUMBER(1) DEFAULT 0 NOT NULL,
    orthopedist NUMBER(1) DEFAULT 0 NOT NULL,
    gastro NUMBER(1) DEFAULT 0 NOT NULL,
    otolaryngologist NUMBER(1) DEFAULT 0 NOT NULL,
);

INSERT INTO Patient (id_number, name, cpf, rg, date_of_birth, agree) VALUES ('012345', 'Bruna', '0', '0', date'1992-12-22', 'Pleno');