# TeamOne Patient App API Documentation

## Base URL
```
https://dispatchv2.dev.team1.sa/api/v1/patient/
```

## Authentication API

### Send OTP
**POST** `/auth/otp/send`

Send OTP code to user's phone number for login.

**Parameters:**
- `phone` (string): User's phone number

### Verify OTP
**POST** `/auth/otp/verify`

Verify the OTP code sent to user's phone.

**Parameters:**
- `phone` (string): User's phone number
- `code` (string): OTP code received
- `device_type` (string): Device type (default: "android")
- `device_token` (string): Device token for push notifications (optional)

### Complete User Information
**POST** `/auth/complete`

Complete user registration with personal and medical information.

**Parameters:**
- `personal` (string): Personal ID
- `patient_name` (string): Patient's full name
- `email` (string): Email address
- `blood_type` (int): Blood type ID
- `gender` (int): Gender ID
- `nationality` (string): Nationality
- `date_of_birth` (string): Birth date
- `medical_history` (string): Medical history details
- `phone` (string): Phone number

## Main Service API

### Create New Case
**POST** `/car_requests/createCase`

Create a comprehensive medical case with detailed patient information.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `type` (int): Case type (optional)
- `transportation_type_id` (int): Transportation type ID (optional)
- `start_date` (string): Start date and time (optional)
- `services` (Map): Service details
- `patient_name` (string): Patient name (optional)
- `personal` (string): Personal ID (optional)
- `gender` (int): Gender ID (optional)
- `phone` (string): Phone number (optional)
- `nationality` (string): Nationality (optional)
- `blood_type` (int): Blood type ID (optional)
- `date_of_birth` (string): Birth date (optional)
- `weight` (string): Patient weight (optional)
- `location` (string): Pickup location (optional)
- `address` (string): Pickup address (optional)
- `address_latitude` (double): Pickup latitude (optional)
- `address_longitude` (double): Pickup longitude (optional)
- `destination` (string): Destination location (optional)
- `destination_latitude` (double): Destination latitude (optional)
- `destination_longitude` (double): Destination longitude (optional)
- `description` (string): Case description (optional)
- `breath` (int): Breathing status (optional)
- `conscious` (int): Consciousness level (optional)
- `contagious_disease` (string): Contagious disease info (optional)
- `medical_history` (string): Medical history (optional)
- `report` (string): Medical report (optional)

### Get Transportation Types
**GET** `/transportation-types`

Retrieve available transportation types.

**Headers:**
- `Authorization`: Bearer token

### Get Transportation Services
**GET** `/services`

Retrieve available transportation services.

**Headers:**
- `Authorization`: Bearer token

### Get Emergency Services
**GET** `/car_requests/emergency/services`

Retrieve available emergency services.

**Headers:**
- `Authorization`: Bearer token

### Add Patient
**POST** `/add`

Add a new patient to the system.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `personal` (string): Personal ID
- `patient_name` (string): Patient name
- `email` (string): Email address
- `phone` (string): Phone number
- `blood_type` (int): Blood type ID
- `gender` (int): Gender ID
- `nationality` (string): Nationality
- `date_of_birth` (string): Birth date
- `medical_history` (string): Medical history

### Review Order
**POST** `/services/price`

Get pricing information for a service request.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `origin_lat` (double): Origin latitude
- `origin_lng` (double): Origin longitude
- `destination_lat` (double): Destination latitude (optional)
- `destination_lng` (double): Destination longitude (optional)
- `service` (int): Service ID

### Get Payment Status
**POST** `/payment/status`

Check payment status for a transaction.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `resource_path` (string): Payment resource path

## Orders API

### Get Orders History
**GET** `/car-request`

Retrieve user's order history.

**Headers:**
- `Authorization`: Bearer token

### Get Order Status
**POST** `/car-request/status`

Get current status of a specific order.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `car_request` (int): Order ID

### Get Invoice
**GET** `/invoice`

Retrieve invoice for a completed order.

**Headers:**
- `Authorization`: Bearer token

**Query Parameters:**
- `car_request_id` (string): Order ID

### Cancel Car Request
**POST** `/car-request/cancel`

Cancel an existing car request.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `car_request_id` (string): Request ID to cancel
- `cancel_request_reason_id` (int): Cancellation reason ID
- `cancel_request_reason` (string): Cancellation reason description

### Submit Review
**POST** `/car-request/rate`

Submit rating and review for a completed service.

**Headers:**
- `Authorization`: Bearer token

**Parameters:**
- `car_request_id` (string): Request ID to rate
- `rate` (string): Rating value
- `comment` (string): Review comment

## Request Status Codes

- `0`: Pending
- `1`: Notified Team
- `2`: Accepted Case
- `3`: Team Moving to Pickup Location
- `4`: Team Arrived to Pickup Location
- `5`: Team Started Examination
- `6`: Team Moving to Destination Location
- `7`: Team Arrived to Destination Location
- `8`: Team Delivered Patient
- `9`: Leaving Final Destination
- `10`: Closed Request
- `11`: Cancelled Request

## Request Types

- `0`: Transportation Type
- `2`: Emergency Type
```