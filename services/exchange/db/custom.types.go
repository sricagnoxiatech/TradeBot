package db

import (
	"database/sql/driver"
	"encoding/json"
)

type CommonStrategyProps struct {
	Enabled bool `json:"enabled"`
}

type Rsi struct {
	CommonStrategyProps
	Period     int `json:"period"`
	Overbought int `json:"overbought"`
	Oversold   int `json:"oversold"`
}

func (Rsi) GormDataType() string {
	return "JSONB"
}

func (r *Rsi) Scan(value any) error {
	return json.Unmarshal(value.([]byte), &r)
}

func (r Rsi) Value() (driver.Value, error) {
	return json.Marshal(r)
}

type Macd struct {
	CommonStrategyProps
	Fast   int `json:"fast"`
	Slow   int `json:"slow"`
	Signal int `json:"signal"`
}

func (Macd) GormDataType() string {
	return "JSONB"
}

func (r *Macd) Scan(value any) error {
	return json.Unmarshal(value.([]byte), &r)
}

func (r Macd) Value() (driver.Value, error) {
	return json.Marshal(r)
}

type Lstm struct {
    CommonStrategyProps
    SequenceLength  int      `json:"sequence_length"`
    PredictionSteps int      `json:"prediction_steps"`
    Units          int      `json:"units"`
    Features       []string `json:"features"`
}

func (Lstm) GormDataType() string {
    return "JSONB"
}

func (l *Lstm) Scan(value any) error {
    return json.Unmarshal(value.([]byte), &l)
}

func (l Lstm) Value() (driver.Value, error) {
    return json.Marshal(l)
}

type Strategies struct {
    Symbol string `gorm:"primaryKey" json:"symbol"`
    Rsi    Rsi    `gorm:"not null" json:"rsi"`
    Macd   Macd   `gorm:"not null" json:"macd"`
    Lstm   Lstm   `gorm:"not null" json:"lstm"`
}