-- Reset the valid_email field
DELIMITER //
DROP TRIGGER IF EXISTS reset_validation;

CREATE TRIGGER reset_validation
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
	IF OLD.email <> NEW.email  THEN
		SET NEW.valid_email = 0;
	END IF;
END
//

DELIMITER ;
