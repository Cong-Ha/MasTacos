-- MySQL script to populate tables with mock data
-- For Ma's Tacos Restaurant Management System

USE mas_tacos;

-- Clear existing data if needed
-- DELETE FROM SurveyResponses;
-- DELETE FROM OrderItems;
-- DELETE FROM Orders;
-- DELETE FROM Reservations;
-- DELETE FROM MenuItems;
-- DELETE FROM TimeSlots;
-- DELETE FROM Customers;

-- Insert data into Customers table
INSERT INTO Customers (FirstName, LastName, Email, Phone, MarketingOptIn, JoinDate, LoyaltyPoints) VALUES
('Maria', 'Garcia', 'maria.garcia@email.com', '555-123-4567', TRUE, '2024-01-15', 120),
('John', 'Smith', 'john.smith@email.com', '555-234-5678', TRUE, '2024-01-20', 85),
('Emily', 'Johnson', 'emily.j@email.com', '555-345-6789', FALSE, '2024-02-03', 50),
('Miguel', 'Rodriguez', 'miguel.r@email.com', '555-456-7890', TRUE, '2024-02-10', 175),
('Sarah', 'Williams', 'sarah.w@email.com', '555-567-8901', TRUE, '2024-02-15', 95),
('David', 'Brown', 'david.b@email.com', '555-678-9012', FALSE, '2024-02-28', 30),
('Lisa', 'Miller', 'lisa.m@email.com', '555-789-0123', TRUE, '2024-03-05', 140),
('James', 'Davis', 'james.d@email.com', '555-890-1234', TRUE, '2024-03-12', 70),
('Jennifer', 'Martinez', 'jennifer.m@email.com', '555-901-2345', FALSE, '2024-03-20', 25),
('Robert', 'Hernandez', 'robert.h@email.com', '555-012-3456', TRUE, '2024-03-28', 110),
('Patricia', 'Lopez', 'patricia.l@email.com', '555-123-7890', TRUE, '2024-04-02', 60),
('Michael', 'Wilson', 'michael.w@email.com', '555-234-8901', FALSE, '2024-04-10', 40),
('Linda', 'Anderson', 'linda.a@email.com', '555-345-9012', TRUE, '2024-04-18', 130),
('William', 'Thomas', 'william.t@email.com', '555-456-0123', TRUE, '2024-04-25', 80),
('Elizabeth', 'Jackson', 'elizabeth.j@email.com', '555-567-1234', FALSE, '2024-05-03', 20),
('Richard', 'White', 'richard.w@email.com', '555-678-2345', TRUE, '2024-05-12', 105),
('Susan', 'Harris', 'susan.h@email.com', '555-789-3456', TRUE, '2024-05-20', 55),
('Joseph', 'Clark', 'joseph.c@email.com', '555-890-4567', FALSE, '2024-05-28', 35),
('Jessica', 'Lewis', 'jessica.l@email.com', '555-901-5678', TRUE, '2024-06-05', 125),
('Thomas', 'Robinson', 'thomas.r@email.com', '555-012-6789', TRUE, '2024-06-15', 75),
('Nancy', 'Walker', 'nancy.w@email.com', '555-123-7890', FALSE, '2024-06-22', 15),
('Christopher', 'Young', 'chris.y@email.com', '555-234-8901', TRUE, '2024-06-30', 95),
('Karen', 'Allen', 'karen.a@email.com', '555-345-9012', TRUE, '2024-07-08', 45),
('Daniel', 'King', 'daniel.k@email.com', '555-456-0123', FALSE, '2024-07-15', 25),
('Betty', 'Wright', 'betty.w@email.com', '555-567-1234', TRUE, '2024-07-23', 115),
('Matthew', 'Scott', 'matthew.s@email.com', '555-678-2345', TRUE, '2024-07-31', 65),
('Dorothy', 'Green', 'dorothy.g@email.com', '555-789-3456', FALSE, '2024-08-08', 10),
('Anthony', 'Baker', 'anthony.b@email.com', '555-890-4567', TRUE, '2024-08-16', 85),
('Sandra', 'Adams', 'sandra.a@email.com', '555-901-5678', TRUE, '2024-08-24', 40),
('Mark', 'Nelson', 'mark.n@email.com', '555-012-6789', FALSE, '2024-09-01', 30);

-- Insert data into TimeSlots table
INSERT INTO TimeSlots (StartTime, EndTime, DayOfWeek, AvgCustomerVolume, PeakHours) VALUES
('11:00:00', '11:30:00', 'Monday', 15, FALSE),
('11:30:00', '12:00:00', 'Monday', 25, FALSE),
('12:00:00', '12:30:00', 'Monday', 40, TRUE),
('12:30:00', '13:00:00', 'Monday', 45, TRUE),
('13:00:00', '13:30:00', 'Monday', 30, FALSE),
('17:00:00', '17:30:00', 'Monday', 20, FALSE),
('17:30:00', '18:00:00', 'Monday', 30, FALSE),
('18:00:00', '18:30:00', 'Monday', 45, TRUE),
('18:30:00', '19:00:00', 'Monday', 50, TRUE),
('19:00:00', '19:30:00', 'Monday', 40, TRUE),
('11:00:00', '11:30:00', 'Friday', 20, FALSE),
('11:30:00', '12:00:00', 'Friday', 35, TRUE),
('12:00:00', '12:30:00', 'Friday', 50, TRUE),
('12:30:00', '13:00:00', 'Friday', 55, TRUE),
('13:00:00', '13:30:00', 'Friday', 40, TRUE),
('17:00:00', '17:30:00', 'Friday', 30, FALSE),
('17:30:00', '18:00:00', 'Friday', 45, TRUE),
('18:00:00', '18:30:00', 'Friday', 60, TRUE),
('18:30:00', '19:00:00', 'Friday', 65, TRUE),
('19:00:00', '19:30:00', 'Friday', 55, TRUE),
('11:00:00', '11:30:00', 'Saturday', 25, FALSE),
('11:30:00', '12:00:00', 'Saturday', 40, TRUE),
('12:00:00', '12:30:00', 'Saturday', 60, TRUE),
('12:30:00', '13:00:00', 'Saturday', 65, TRUE),
('13:00:00', '13:30:00', 'Saturday', 50, TRUE),
('17:00:00', '17:30:00', 'Saturday', 35, TRUE),
('17:30:00', '18:00:00', 'Saturday', 55, TRUE),
('18:00:00', '18:30:00', 'Saturday', 70, TRUE),
('18:30:00', '19:00:00', 'Saturday', 75, TRUE),
('19:00:00', '19:30:00', 'Saturday', 65, TRUE);

-- Insert data into MenuItems table
INSERT INTO MenuItems (Name, Description, Price, Category, IsActive, PopularityScore) VALUES
('Carne Asada Taco', 'Grilled steak taco with cilantro and onions', 4.50, 'Tacos', TRUE, 95),
('Chicken Tinga Taco', 'Shredded chicken in chipotle sauce', 4.00, 'Tacos', TRUE, 85),
('Al Pastor Taco', 'Marinated pork with pineapple', 4.25, 'Tacos', TRUE, 90),
('Vegetarian Taco', 'Grilled vegetables with guacamole', 3.75, 'Tacos', TRUE, 70),
('Fish Taco', 'Battered fish with cabbage slaw and lime crema', 4.75, 'Tacos', TRUE, 80),
('Shrimp Taco', 'Grilled shrimp with mango salsa', 5.00, 'Tacos', TRUE, 75),
('Birria Taco', 'Slow-cooked beef with consomme for dipping', 5.50, 'Tacos', TRUE, 98),
('Chorizo Taco', 'Spicy Mexican sausage with potato', 4.25, 'Tacos', TRUE, 65),
('Bean and Cheese Burrito', 'Refried beans and melted cheese', 7.50, 'Burritos', TRUE, 60),
('Carne Asada Burrito', 'Grilled steak with rice, beans, and salsa', 9.50, 'Burritos', TRUE, 88),
('Chicken Burrito', 'Grilled chicken with rice, beans, and salsa', 8.50, 'Burritos', TRUE, 82),
('Veggie Burrito', 'Grilled vegetables with rice, beans, and guacamole', 7.50, 'Burritos', TRUE, 60),
('Super Burrito', 'Meat, rice, beans, cheese, sour cream, and guacamole', 10.50, 'Burritos', TRUE, 85),
('Nachos Supreme', 'Tortilla chips topped with meat, cheese, beans, and sour cream', 9.00, 'Appetizers', TRUE, 78),
('Guacamole and Chips', 'Fresh guacamole with house-made tortilla chips', 6.50, 'Appetizers', TRUE, 72),
('Queso Fundido', 'Melted cheese with chorizo and tortillas', 7.00, 'Appetizers', TRUE, 65),
('Quesadilla', 'Flour tortilla with melted cheese and choice of filling', 6.50, 'Appetizers', TRUE, 70),
('Mexican Rice', 'Traditional tomato-based rice', 2.50, 'Sides', TRUE, 55),
('Refried Beans', 'Pinto beans cooked and mashed with spices', 2.50, 'Sides', TRUE, 50),
('Elote', 'Mexican street corn with mayo, cheese, and chili powder', 3.50, 'Sides', TRUE, 75),
('Horchata', 'Sweet rice milk with cinnamon', 3.00, 'Beverages', TRUE, 72),
('Jamaica', 'Hibiscus tea', 3.00, 'Beverages', TRUE, 65),
('Mexican Coca-Cola', 'Made with real sugar', 3.50, 'Beverages', TRUE, 80),
('Jarritos', 'Mexican fruit soda, various flavors', 3.00, 'Beverages', TRUE, 68),
('Churros', 'Fried dough pastry with cinnamon sugar', 4.00, 'Desserts', TRUE, 82),
('Flan', 'Caramel custard', 4.50, 'Desserts', TRUE, 75),
('Tres Leches Cake', 'Sponge cake soaked in three kinds of milk', 5.00, 'Desserts', TRUE, 80),
('Sopapillas', 'Fried pastry with honey and cinnamon', 4.00, 'Desserts', TRUE, 70),
('Taco Salad', 'Salad in a crispy tortilla bowl with choice of meat', 8.50, 'Entrees', TRUE, 68),
('Enchiladas', 'Corn tortillas filled with meat, topped with sauce and cheese', 9.50, 'Entrees', TRUE, 78);

-- Insert data into Reservations table
INSERT INTO Reservations (CustomerId, ReservationTime, PartySize, Status, SpecialRequests, TimeSlotId) VALUES
(1, '2024-09-15 18:00:00', 2, 'Confirmed', 'Window seat preferred', 18),
(3, '2024-09-16 12:00:00', 4, 'Confirmed', 'Birthday celebration', 3),
(5, '2024-09-17 19:00:00', 3, 'Confirmed', NULL, 10),
(7, '2024-09-18 12:30:00', 2, 'Confirmed', NULL, 4),
(9, '2024-09-19 18:30:00', 6, 'Confirmed', 'High chair needed', 19),
(11, '2024-09-20 13:00:00', 4, 'Confirmed', NULL, 15),
(13, '2024-09-21 18:00:00', 2, 'Confirmed', NULL, 28),
(15, '2024-09-22 12:00:00', 3, 'Confirmed', NULL, 23),
(17, '2024-09-15 19:00:00', 5, 'Completed', 'Anniversary dinner', 20),
(19, '2024-09-16 18:30:00', 4, 'Completed', NULL, 9),
(21, '2024-09-17 12:30:00', 2, 'Completed', NULL, 14),
(23, '2024-09-18 18:00:00', 3, 'Completed', NULL, 8),
(25, '2024-09-19 12:00:00', 4, 'Completed', 'Gluten-free options needed', 3),
(27, '2024-09-20 19:00:00', 2, 'Completed', NULL, 20),
(29, '2024-09-21 12:30:00', 3, 'Completed', NULL, 24),
(2, '2024-09-22 18:30:00', 6, 'Cancelled', 'Changed plans', 29),
(4, '2024-09-15 13:00:00', 2, 'Cancelled', NULL, 5),
(6, '2024-09-16 19:00:00', 4, 'No-Show', NULL, 10),
(8, '2024-09-17 12:00:00', 2, 'Confirmed', NULL, 13),
(10, '2024-09-18 18:30:00', 5, 'Confirmed', 'Corner booth if possible', 9),
(12, '2024-09-19 12:30:00', 3, 'Confirmed', NULL, 4),
(14, '2024-09-20 18:00:00', 4, 'Confirmed', 'Business dinner', 18),
(16, '2024-09-21 13:00:00', 2, 'Confirmed', NULL, 25),
(18, '2024-09-22 19:00:00', 3, 'Confirmed', NULL, 30),
(20, '2024-09-15 12:00:00', 4, 'Confirmed', NULL, 12),
(22, '2024-09-16 18:30:00', 2, 'Confirmed', NULL, 19),
(24, '2024-09-17 12:30:00', 6, 'Confirmed', 'Birthday celebration', 13),
(26, '2024-09-18 19:00:00', 3, 'Confirmed', NULL, 10),
(28, '2024-09-19 18:00:00', 2, 'Confirmed', NULL, 8),
(30, '2024-09-20 12:00:00', 4, 'Confirmed', NULL, 3);

-- Insert data into Orders table
INSERT INTO Orders (CustomerId, OrderTime, TotalAmount, OrderStatus, OrderType, TimeSlotId) VALUES
(1, '2024-09-15 18:15:00', 28.50, 'Completed', 'Dine-In', 18),
(2, '2024-09-15 12:30:00', 22.75, 'Completed', 'Takeout', 3),
(3, '2024-09-16 12:15:00', 37.50, 'Completed', 'Dine-In', 3),
(4, '2024-09-16 18:45:00', 19.25, 'Completed', 'Delivery', NULL),
(5, '2024-09-17 19:10:00', 32.00, 'Completed', 'Dine-In', 10),
(6, '2024-09-17 12:30:00', 25.50, 'Completed', 'Takeout', 13),
(7, '2024-09-18 12:45:00', 21.75, 'Completed', 'Dine-In', 4),
(8, '2024-09-18 19:00:00', 43.25, 'Completed', 'Delivery', NULL),
(9, '2024-09-19 18:40:00', 65.50, 'Completed', 'Dine-In', 19),
(10, '2024-09-19 12:20:00', 31.00, 'Completed', 'Takeout', 3),
(11, '2024-09-20 13:15:00', 37.25, 'Completed', 'Dine-In', 15),
(12, '2024-09-20 18:30:00', 29.75, 'Completed', 'Delivery', NULL),
(13, '2024-09-21 18:10:00', 22.50, 'Completed', 'Dine-In', 28),
(14, '2024-09-21 12:45:00', 41.25, 'Completed', 'Takeout', 23),
(15, '2024-09-22 12:15:00', 28.00, 'Completed', 'Dine-In', 23),
(16, '2024-09-22 18:45:00', 34.50, 'Preparing', 'Delivery', NULL),
(17, '2024-09-15 19:15:00', 52.75, 'Completed', 'Dine-In', 20),
(18, '2024-09-16 12:30:00', 19.25, 'Completed', 'Takeout', 3),
(19, '2024-09-16 18:40:00', 38.50, 'Completed', 'Dine-In', 9),
(20, '2024-09-17 12:20:00', 27.00, 'Completed', 'Delivery', NULL),
(21, '2024-09-17 18:15:00', 24.50, 'Completed', 'Takeout', 8),
(22, '2024-09-18 12:40:00', 19.75, 'Completed', 'Dine-In', 4),
(23, '2024-09-18 18:15:00', 32.25, 'Completed', 'Dine-In', 8),
(24, '2024-09-19 12:10:00', 28.50, 'Completed', 'Takeout', 3),
(25, '2024-09-19 19:05:00', 41.00, 'Completed', 'Dine-In', 20),
(26, '2024-09-20 12:30:00', 23.75, 'Completed', 'Delivery', NULL),
(27, '2024-09-20 18:20:00', 31.50, 'Completed', 'Takeout', 18),
(28, '2024-09-21 12:15:00', 29.25, 'Completed', 'Dine-In', 23),
(29, '2024-09-21 18:45:00', 37.00, 'Preparing', 'Delivery', NULL),
(30, '2024-09-22 12:30:00', 25.50, 'Pending', 'Takeout', 23);

-- Insert data into OrderItems table
INSERT INTO OrderItems (OrderId, MenuItemId, Quantity, UnitPrice, SpecialInstructions) VALUES
(1, 1, 2, 4.50, NULL),
(1, 14, 1, 9.00, NULL),
(1, 21, 2, 3.00, NULL),
(2, 3, 3, 4.25, 'Extra salsa'),
(2, 22, 2, 3.00, NULL),
(3, 7, 2, 5.50, NULL),
(3, 15, 1, 6.50, NULL),
(3, 25, 2, 4.00, NULL),
(4, 2, 2, 4.00, NULL),
(4, 18, 1, 2.50, NULL),
(4, 19, 1, 2.50, NULL),
(5, 10, 1, 9.50, 'No onions'),
(5, 16, 1, 7.00, NULL),
(5, 23, 2, 3.50, NULL),
(6, 5, 3, 4.75, NULL),
(6, 17, 1, 6.50, NULL),
(7, 6, 2, 5.00, 'Extra lime'),
(7, 21, 2, 3.00, NULL),
(8, 11, 1, 8.50, NULL),
(8, 15, 1, 6.50, NULL),
(8, 18, 1, 2.50, NULL),
(8, 19, 1, 2.50, NULL),
(8, 26, 2, 4.50, NULL),
(9, 7, 4, 5.50, NULL),
(9, 14, 2, 9.00, NULL),
(9, 20, 2, 3.50, NULL),
(9, 24, 3, 3.00, NULL),
(10, 12, 1, 7.50, 'Extra guacamole'),
(10, 17, 1, 6.50, NULL),
(10, 27, 2, 5.00, NULL),
(11, 8, 2, 4.25, NULL),
(11, 15, 1, 6.50, NULL),
(11, 18, 1, 2.50, NULL),
(11, 29, 2, 8.50, NULL),
(12, 4, 3, 3.75, 'Extra guacamole'),
(12, 19, 2, 2.50, NULL),
(12, 22, 2, 3.00, NULL),
(13, 1, 2, 4.50, NULL),
(13, 2, 1, 4.00, NULL),
(13, 21, 3, 3.00, NULL),
(14, 13, 1, 10.50, NULL),
(14, 14, 1, 9.00, NULL),
(14, 24, 2, 3.00, NULL),
(14, 26, 2, 4.50, NULL),
(15, 5, 2, 4.75, NULL),
(15, 16, 1, 7.00, NULL),
(15, 20, 2, 3.50, NULL),
(16, 10, 1, 9.50, NULL),
(16, 18, 1, 2.50, NULL),
(16, 19, 1, 2.50, NULL),
(16, 25, 2, 4.00, NULL),
(17, 7, 3, 5.50, NULL),
(17, 15, 2, 6.50, 'Extra chips'),
(17, 23, 3, 3.50, NULL),
(17, 27, 2, 5.00, NULL),
(18, 3, 2, 4.25, NULL),
(18, 18, 1, 2.50, NULL),
(18, 22, 1, 3.00, NULL),
(19, 11, 2, 8.50, NULL),
(19, 14, 1, 9.00, NULL),
(19, 24, 2, 3.00, NULL),
(20, 6, 3, 5.00, NULL),
(20, 19, 2, 2.50, NULL),
(21, 2, 2, 4.00, NULL),
(21, 17, 1, 6.50, NULL),
(21, 21, 2, 3.00, NULL),
(22, 4, 2, 3.75, 'Extra veggies'),
(22, 18, 1, 2.50, NULL),
(22, 19, 1, 2.50, NULL),
(23, 13, 1, 10.50, NULL),
(23, 20, 2, 3.50, NULL),
(23, 25, 2, 4.00, NULL),
(24, 8, 3, 4.25, NULL),
(24, 15, 1, 6.50, NULL),
(24, 24, 2, 3.00, NULL),
(25, 7, 2, 5.50, NULL),
(25, 14, 1, 9.00, NULL),
(25, 26, 2, 4.50, NULL),
(25, 21, 2, 3.00, NULL),
(26, 3, 2, 4.25, NULL),
(26, 18, 1, 2.50, NULL),
(26, 19, 1, 2.50, NULL),
(26, 22, 2, 3.00, NULL),
(27, 10, 1, 9.50, NULL),
(27, 16, 1, 7.00, NULL),
(27, 23, 3, 3.50, NULL),
(28, 5, 2, 4.75, 'Extra slaw'),
(28, 17, 1, 6.50, NULL),
(28, 25, 2, 4.00, NULL),
(29, 11, 2, 8.50, NULL),
(29, 14, 1, 9.00, NULL),
(29, 20, 2, 3.50, NULL),
(30, 2, 2, 4.00, NULL),
(30, 15, 1, 6.50, NULL),
(30, 21, 2, 3.00, NULL);

-- Insert data into SurveyResponses table
INSERT INTO SurveyResponses (CustomerId, OrderId, SubmissionDate, FoodRating, ServiceRating, AmbienceRating, Feedback, FollowedUp) VALUES
(1, 1, '2024-09-15 19:30:00', 5, 4, 4, 'Great tacos! Service was quick and friendly.', FALSE),
(2, 2, '2024-09-15 13:15:00', 4, 5, NULL, 'Enjoyed the Al Pastor tacos. Will order again.', FALSE),
(3, 3, '2024-09-16 13:45:00', 5, 5, 5, 'The birria tacos were amazing! Perfect for our celebration.', TRUE),
(4, 4, '2024-09-16 19:30:00', 3, 4, NULL, 'Food was good but delivery took longer than expected.', TRUE),
(5, 5, '2024-09-17 20:30:00', 5, 4, 4, 'Loved the burrito and queso fundido. Will be back!', FALSE),
(6, 6, '2024-09-17 13:15:00', 4, 4, NULL, 'Fish tacos were fresh and delicious.', FALSE),
(7, 7, '2024-09-18 14:00:00', 5, 5, 4, 'Shrimp tacos were perfect. Loved the horchata too!', FALSE),
(8, 8, '2024-09-18 20:15:00', 4, 3, NULL, 'Good food but some items were missing from delivery.', TRUE),
(9, 9, '2024-09-19 20:00:00', 5, 5, 5, 'Fantastic meal for our family. The kids loved the churros!', FALSE),
(10, 10, '2024-09-19 13:30:00', 4, 4, NULL, 'Veggie burrito was great. Would like more sauce options.', FALSE),
(11, 11, '2024-09-20 14:30:00', 5, 4, 4, 'Chorizo tacos had amazing flavor. Will recommend to friends!', FALSE),
(12, 12, '2024-09-20 19:45:00', 3, 4, NULL, 'Vegetarian tacos were good but could use more seasoning.', TRUE),
(13, 13, '2024-09-21 19:30:00', 5, 5, 5, 'Perfect date night meal. Loved the ambience.', FALSE),
(14, 14, '2024-09-21 13:45:00', 4, 4, NULL, 'Super burrito was huge and delicious!', FALSE),
(15, 15, '2024-09-22 13:30:00', 5, 4, 4, 'Fish tacos were fresh and the elote was perfect.', FALSE),
(17, 17, '2024-09-15 20:45:00', 5, 5, 5, 'Our anniversary dinner was wonderful. Thank you!', TRUE),
(18, 18, '2024-09-16 13:15:00', 4, 4, NULL, 'Solid Al Pastor tacos. Will order again.', FALSE),
(19, 19, '2024-09-16 20:00:00', 5, 4, 4, 'The chicken burritos were a hit with the family!', FALSE),
(20, 20, '2024-09-