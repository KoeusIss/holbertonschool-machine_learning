-- Creates a trigger that decrease the quantity
-- Of an item after adding a new order
CREATE TRIGGER UpdateItems
AFTER INSERT ON orders
FOR EACH ROW
	UPDATE items
	SET items.quantity = items.quantity - NEW.number
	WHERE items.name = NEW.item_name;
